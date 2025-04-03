import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from ..models import CoGraphNet
from .utils import (
    setup_logger, compute_ece, plot_reliability_diagram, 
    plot_prediction_distribution, plot_class_distribution,
    plot_unique_classes_per_batch
)
from .focal_loss import FocalLoss

class CoGraphTrainer:
    def __init__(
        self,
        model: CoGraphNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        rank: int = 0,
        world_size: int = 1,
        num_epochs: int = 100,
        train_class_weights: Optional[torch.Tensor] = None,
        val_class_weights: Optional[torch.Tensor] = None,
        test_class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        plot_dir: str = "plots",
        num_classes: int = 10
    ):
        self.logger = setup_logger()
        self.rank = rank
        self.world_size = world_size
        self.num_epochs = num_epochs
        self.train_class_weights = train_class_weights
        self.val_class_weights = val_class_weights
        self.test_class_weights = test_class_weights
        self.plot_dir = plot_dir
        self.num_classes = num_classes
        # Create plot directory if it doesn't exist
        if self.rank == 0 and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Model
        self.model = model
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Training components
        self.train_criterion = FocalLoss(gamma=gamma, weight=self.train_class_weights)
        self.val_criterion = FocalLoss(gamma=gamma, weight=self.val_class_weights)
        self.test_criterion = FocalLoss(gamma=gamma, weight=self.test_class_weights)
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5, verbose=True)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.batch_indices = []
        self.batch_unique_pred = []
        self.batch_unique_true = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _gather_metrics(self, loss: float, samples: int, correct: int = 0) -> tuple[float, int, int]:
        """Gather metrics from all processes."""
        # Convert to tensors
        loss_tensor = torch.tensor([loss])
        samples_tensor = torch.tensor([samples])
        correct_tensor = torch.tensor([correct])
        
        # Gather from all processes
        all_losses = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
        all_samples = [torch.zeros_like(samples_tensor) for _ in range(self.world_size)]
        all_corrects = [torch.zeros_like(correct_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(all_losses, loss_tensor)
        dist.all_gather(all_samples, samples_tensor)
        dist.all_gather(all_corrects, correct_tensor)
        
        # Sum across processes
        total_loss = sum(l.item() for l in all_losses) / self.world_size
        total_samples = sum(s.item() for s in all_samples)
        total_correct = sum(c.item() for c in all_corrects)
        
        return total_loss, total_samples, total_correct
    
    def freeze_all_except_sentence(self):
        """Freeze all layers except the Sentence Model."""
        for param in self.model.module.word_net.parameters():
            param.requires_grad = False
        for param in self.model.module.fusion.parameters():
            param.requires_grad = False
        for param in self.model.module.sent_net.parameters():
            param.requires_grad = True
        for param in self.model.module.final_mlp.parameters():
            param.requires_grad = True
        self.logger.info("Training Sentence Model only.")

    def freeze_all_except_word(self):
        """Freeze all layers except the Word Model."""
        for param in self.model.module.sent_net.parameters():
            param.requires_grad = False
        for param in self.model.module.fusion.parameters():
            param.requires_grad = False
        for param in self.model.module.word_net.parameters():
            param.requires_grad = True
        for param in self.model.module.final_mlp.parameters():
            param.requires_grad = True
        self.logger.info("Training Word Model only.")

    def freeze_all_except_fusion(self):
        """Freeze all layers except the Fusion Layer."""
        for param in self.model.module.word_net.parameters():
            param.requires_grad = False
        for param in self.model.module.sent_net.parameters():
            param.requires_grad = False
        for param in self.model.module.fusion.parameters():
            param.requires_grad = True
        for param in self.model.module.final_mlp.parameters():
            param.requires_grad = True
        self.logger.info("Training Fusion Layer only.")

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.module.parameters():
            param.requires_grad = True
        self.logger.info("Fine-tuning all layers.")
        
    def train_epoch(self, epoch: int, phase: int) -> float:
        self.model.train()
        total_loss = 0
        total_samples = 0

        accumulation_steps = 1  # Number of steps to accumulate gradients before updating
        accumulated_loss = 0  # Track accumulated loss
        
        # Reset batch tracking
        self.batch_indices = []
        self.batch_unique_pred = []
        self.batch_unique_true = []
        
        with tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}',
            disable=self.rank != 0,
            total=len(self.train_loader)  # Explicitly set total
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                
                # Extract word graph components:
                word_x = batch['word'].x.to(self.device)
                word_edge_index = batch['word', 'co_occurs', 'word'].edge_index.to(self.device)
                word_edge_weight = batch['word', 'co_occurs', 'word'].edge_attr.to(self.device) if 'edge_attr' in batch['word', 'co_occurs', 'word'] else None
                word_batch = batch['word'].batch.to(self.device)
                
                # Extract sentence graph components:
                sent_x = batch['sentence'].x.to(self.device)
                sent_edge_index = batch['sentence', 'related_to', 'sentence'].edge_index.to(self.device)
                sent_edge_weight = batch['sentence', 'related_to', 'sentence'].edge_attr.to(self.device) if 'edge_attr' in batch['sentence', 'related_to', 'sentence'] else None
                sent_batch = batch['sentence'].batch.to(self.device)
                
                # Forward pass: call the model with separate word and sentence subgraphs.
                outputs = self.model(
                    word_x, word_edge_index, word_batch, word_edge_weight,
                    sent_x, sent_edge_index, sent_batch, sent_edge_weight
                )

                batch.y = batch.y.to(torch.long)
                batch_size = batch.y.size(0)
                
                loss = self.train_criterion(outputs[:batch_size], batch.y[:batch_size])
                loss = loss / accumulation_steps  # Scale loss for accumulation
                # Break after printting
                # WHen loss is nan print outputs and any nan gradients if they exist
                # Outputs are nan for bad batch, lets check data for word, sentence and edge attributes
                if torch.isnan(loss).any():
                    print(f"Loss is NaN for batch {batch_idx}")
                    print(f"Outputs: {outputs[:batch_size]}")
                    print(f"Batch: {batch}")
                    print(f"Batch word x: {word_x}")
                    print(f"Batch sentence x: {sent_x}")
                    print(f"Batch word edge index: {word_edge_index}")
                    print(f"Batch sentence edge index: {sent_edge_index}")
                    print(f"Batch word edge weight: {word_edge_weight}")
                    print(f"Batch sentence edge weight: {sent_edge_weight}")
                    print(f"Batch size: {batch_size}")
                    print(f"Batch y: {batch.y[:batch_size]}")
                    break
                
                # Track unique classes
                preds = outputs[:batch_size].argmax(dim=1)
                unique_pred = len(torch.unique(preds).cpu().numpy())
                unique_true = len(torch.unique(batch.y[:batch_size]).cpu().numpy())
                self.batch_indices.append(batch_idx + 1)
                self.batch_unique_pred.append(unique_pred)
                self.batch_unique_true.append(unique_true)
                
                # Backward pass
                loss.backward()

                # Check for nan values in gradients for gradients that exist
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"Gradient is NaN for parameter: {param.name}")
                        print(f"Gradient: {param.grad}")
                        print(f"Parameter: {param}")
                        print(f"Parameter shape: {param.shape}")
                        break

                # Accumulate loss for tracking
                accumulated_loss += loss.item()
                total_samples += batch_size

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Perform optimizer step only every `accumulation_steps`
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.distributed.barrier()  # Ensure all processes reach this point before reducing gradients
                    # Print accumulated gradients
                    # Check for nan values in gradients for gradients that exist
                    for param in self.model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"Gradient is NaN for parameter: {param.name}")
                            print(f"Gradient: {param.grad}")
                            print(f"Parameter: {param}")
                            print(f"Parameter shape: {param.shape}")
                            print(f"Parameter requires_grad: {param.requires_grad}")
                            break

                    self.optimizer.step()
                    self.optimizer.zero_grad()  # Clear accumulated gradients

                    # Accumulate loss across all processes
                    loss_tensor = torch.tensor([accumulated_loss], device=self.device)
                    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                    total_loss += loss_tensor.item()  # Accumulate the reduced loss

                    torch.distributed.barrier()
                    for param in self.model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"Gradient is NaN for parameter: {param.name}")
                            print(f"Gradient: {param.grad}")
                            print(f"Parameter: {param}")
                            print(f"Parameter shape: {param.shape}")
                            print(f"Parameter requires_grad: {param.requires_grad}")
                            break

                    accumulated_loss = 0  # Reset accumulated loss after step
                
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'epoch': f'{epoch}/{self.num_epochs}'  # Add epoch counter
                    })
        
        torch.distributed.barrier()
        # Gather metrics from all processes
        # Divide total loss by number of batches
        total_loss = total_loss / len(self.train_loader)
        total_loss, total_samples, _ = self._gather_metrics(total_loss, total_samples)
        torch.distributed.barrier()
        
        # Plot unique classes per batch on rank 0
        if self.rank == 0:
            plot_unique_classes_per_batch(
                self.batch_indices,
                self.batch_unique_pred,
                self.batch_unique_true,
                self.num_classes,
                phase,
                epoch,
                self.plot_dir
            )
        
        return total_loss
    
    def validate(self, epoch: int, phase: int) -> Tuple[float, float]:
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extract subgraphs as in training:
                word_x = batch['word'].x.to(self.device)
                word_edge_index = batch['word', 'co_occurs', 'word'].edge_index.to(self.device)
                word_edge_weight = batch['word', 'co_occurs', 'word'].edge_attr.to(self.device) if 'edge_attr' in batch['word', 'co_occurs', 'word'] else None
                word_batch = batch['word'].batch.to(self.device)
                
                sent_x = batch['sentence'].x.to(self.device)
                sent_edge_index = batch['sentence', 'related_to', 'sentence'].edge_index.to(self.device)
                sent_edge_weight = batch['sentence', 'related_to', 'sentence'].edge_attr.to(self.device) if 'edge_attr' in batch['sentence', 'related_to', 'sentence'] else None
                sent_batch = batch['sentence'].batch.to(self.device)
                
                outputs = self.model(
                    word_x, word_edge_index, word_batch, word_edge_weight,
                    sent_x, sent_edge_index, sent_batch, sent_edge_weight
                )

                batch = batch.to(self.device)
                batch_size = batch.y.size(0)
                
                loss = self.val_criterion(outputs[:batch_size], batch.y[:batch_size])
                
                # Calculate accuracy
                pred = torch.softmax(outputs[:batch_size], dim=1).argmax(dim=1)
                correct += pred.eq(batch.y[:batch_size]).sum().item()
                
                # Store predictions and labels for plotting
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y[:batch_size].cpu().numpy())
                all_logits.append(outputs[:batch_size].cpu())
                
                # Update metrics
                total_loss += loss.item()
                total_samples += batch_size
        
        torch.distributed.barrier()
        # Gather metrics from all processes
        # Divide total loss by number of batches
        total_loss = total_loss / len(self.val_loader)
        total_loss, total_samples, correct = self._gather_metrics(
            total_loss, total_samples, correct
        )
        torch.distributed.barrier()
        # Plot validation metrics on rank 0
        if self.rank == 0:
            # Concatenate logits and compute probabilities
            all_logits = torch.cat(all_logits, dim=0)
            all_probs = torch.softmax(all_logits, dim=1).numpy()
            all_labels_np = np.array(all_labels)
            
            # Plot prediction distribution
            plot_prediction_distribution(
                all_preds,
                {str(i): i for i in range(self.num_classes)},
                phase,
                epoch,
                self.plot_dir
            )
            
            # Plot class distribution
            plot_class_distribution(
                all_preds,
                all_labels,
                {str(i): i for i in range(self.num_classes)},
                phase,
                epoch,
                self.plot_dir
            )
            
            # Plot reliability diagram
            plot_reliability_diagram(
                all_probs,
                all_labels_np,
                n_bins=10,
                save_path=os.path.join(self.plot_dir, f"reliability_diagram_phase_{phase}_epoch_{epoch+1}.png")
            )
            
            # Log calibration metrics
            ece = compute_ece(all_probs, all_labels_np)
            self.logger.info(f"[Phase {phase}, Epoch {epoch+1}] Expected Calibration Error (ECE): {ece:.4f}")
        
        return total_loss, correct / total_samples
    
    def test(self) -> Tuple[float, float]:
        """Test the model on the test set."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                word_x = batch['word'].x.to(self.device)
                word_edge_index = batch['word', 'co_occurs', 'word'].edge_index.to(self.device)
                word_edge_weight = batch['word', 'co_occurs', 'word'].edge_attr.to(self.device) if 'edge_attr' in batch['word', 'co_occurs', 'word'] else None
                word_batch = batch['word'].batch.to(self.device)
                
                sent_x = batch['sentence'].x.to(self.device)
                sent_edge_index = batch['sentence', 'related_to', 'sentence'].edge_index.to(self.device)
                sent_edge_weight = batch['sentence', 'related_to', 'sentence'].edge_attr.to(self.device) if 'edge_attr' in batch['sentence', 'related_to', 'sentence'] else None
                sent_batch = batch['sentence'].batch.to(self.device)
                
                outputs = self.model(
                    word_x, word_edge_index, word_batch, word_edge_weight,
                    sent_x, sent_edge_index, sent_batch, sent_edge_weight
                )

                batch = batch.to(self.device)
                batch_size = batch.y.size(0)
                
                loss = self.test_criterion(outputs[:batch_size], batch.y[:batch_size])
                
                # Calculate accuracy
                pred = torch.softmax(outputs[:batch_size], dim=1).argmax(dim=1)
                correct += pred.eq(batch.y[:batch_size]).sum().item()
                
                # Update metrics
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Gather metrics from all processes
        # Divide total loss by number of batches
        total_loss = total_loss / len(self.test_loader)
        torch.distributed.barrier()
        total_loss, total_samples, correct = self._gather_metrics(
            total_loss, total_samples, correct
        )
        torch.distributed.barrier()
        return total_loss, correct / total_samples
        
        # Final test if test loader provided
        if self.test_loader:
            test_loss, test_acc = self.test()
            self.logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}') 
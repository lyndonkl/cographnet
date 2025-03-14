import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models import CoGraphNet
from .metrics import calculate_metrics
from .utils import setup_logger
from .focal_loss import FocalLoss

class CoGraphTrainer:
    def __init__(
        self,
        model: CoGraphNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        rank: int = 0,
        world_size: int = 1,
        num_epochs: int = 100,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.logger = setup_logger()
        self.rank = rank
        self.world_size = world_size
        self.num_epochs = num_epochs
        self.class_weights = class_weights
        
        # Model
        self.model = model
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Training components
        self.criterion = FocalLoss(gamma=2.0, weight=self.class_weights)
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5, verbose=True)
        
        # Tracking
        self.best_val_loss = float('inf')

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
        total_loss = sum(l.item() for l in all_losses)
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
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        total_samples = 0

        accumulation_steps = 64  # Number of steps to accumulate gradients before updating
        accumulated_loss = 0  # Track accumulated loss
        
        
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
                
                loss = self.criterion(outputs[:batch_size], batch.y[:batch_size])
                loss = loss / accumulation_steps  # Scale loss for accumulation
                
                # Backward pass
                loss.backward()

                # Accumulate loss for tracking
                accumulated_loss += loss.item()
                total_samples += batch_size

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Perform optimizer step only every `accumulation_steps`
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.distributed.barrier()  # Ensure all processes reach this point before reducing gradients

                    self.optimizer.step()
                    self.optimizer.zero_grad()  # Clear accumulated gradients

                    # Accumulate loss across all processes
                    loss_tensor = torch.tensor([accumulated_loss], device=self.device)
                    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                    total_loss += loss_tensor.item()  # Accumulate the reduced loss

                    torch.distributed.barrier()

                    accumulated_loss = 0  # Reset accumulated loss after step
                
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'epoch': f'{epoch}/{self.num_epochs}'  # Add epoch counter
                    })
        
        # Gather metrics from all processes
        total_loss, total_samples, _ = self._gather_metrics(total_loss, total_samples)
        
        return total_loss / total_samples
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        
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
                
                loss = self.criterion(outputs[:batch_size], batch.y[:batch_size])
                
                # Calculate accuracy
                pred = torch.softmax(outputs[:batch_size], dim=1).argmax(dim=1)
                correct += pred.eq(batch.y[:batch_size]).sum().item()
                
                # Update metrics
                total_loss += loss.item()
                total_samples += batch_size
        
        # Gather metrics from all processes
        total_loss, total_samples, correct = self._gather_metrics(
            total_loss, total_samples, correct
        )
        
        return total_loss / total_samples, correct / total_samples
    
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
                
                loss = self.criterion(outputs[:batch_size], batch.y[:batch_size])
                
                # Calculate accuracy
                pred = torch.softmax(outputs[:batch_size], dim=1).argmax(dim=1)
                correct += pred.eq(batch.y[:batch_size]).sum().item()
                
                # Update metrics
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Gather metrics from all processes
        total_loss, total_samples, correct = self._gather_metrics(
            total_loss, total_samples, correct
        )
        
        return total_loss / total_samples, correct / total_samples
        
        # Final test if test loader provided
        if self.test_loader:
            test_loss, test_acc = self.test()
            self.logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}') 
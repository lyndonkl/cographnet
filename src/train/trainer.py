import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models import CoGraphNet
from .metrics import calculate_metrics
from .utils import setup_logger

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
        num_epochs: int = 100
    ):
        self.logger = setup_logger()
        self.rank = rank
        self.world_size = world_size
        self.num_epochs = num_epochs
        
        # Model
        self.model = model
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
        
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
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        
        with tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}',
            disable=self.rank != 0,
            total=len(self.train_loader)  # Explicitly set total
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                print(f"\nBatch {batch_idx} shapes:")
                print(f"Batch type: {type(batch)}")
                print(f"Batch keys: {batch.keys()}")
                print(f"Word features shape: {batch['word'].x.shape}")
                print(f"Word batch indices shape: {batch['word'].batch.shape}")
                print(f"Sentence features shape: {batch['sentence'].x.shape}")
                print(f"Sentence batch indices shape: {batch['sentence'].batch.shape}")
                print(f"Target shape before device: {batch.y.shape}")
                
                batch = batch.to(self.device)
                batch_size = batch.y.size(0)
                print(f"Batch size from target: {batch_size}")
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)  # Should be [batch_size, num_classes]
                print(f"Model output shape: {outputs.shape}")
                print(f"Target shape: {batch.y.shape}")
                
                loss = self.criterion(outputs[:batch_size], batch.y[:batch_size])
                print(f"Loss value: {loss.item():.4f}")
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
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
                batch = batch.to(self.device)
                batch_size = batch.y.size(0)
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs[:batch_size], batch.y[:batch_size])
                
                # Calculate accuracy
                pred = outputs[:batch_size].argmax(dim=1)
                correct += pred.eq(batch.y[:batch_size]).sum().item()
                
                # Update metrics
                total_loss += loss.item() * batch_size
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
                batch = batch.to(self.device)
                batch_size = batch.y.size(0)
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs[:batch_size], batch.y[:batch_size])
                
                # Calculate accuracy
                pred = outputs[:batch_size].argmax(dim=1)
                correct += pred.eq(batch.y[:batch_size]).sum().item()
                
                # Update metrics
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Gather metrics from all processes
        total_loss, total_samples, correct = self._gather_metrics(
            total_loss, total_samples, correct
        )
        
        return total_loss / total_samples, correct / total_samples
    
    def train(self, num_epochs: int, save_path: str = 'checkpoints'):
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Log progress
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_acc:.4f}'
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_path, 'best_model.pt')
                )
                self.logger.info(f'New best model saved at epoch {epoch}')
            
            # Update learning rate
            self.scheduler.step()
        
        # Final test if test loader provided
        if self.test_loader:
            test_loss, test_acc = self.test()
            self.logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}') 
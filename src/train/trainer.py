import os
from typing import Dict, Optional

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
        world_size: int = 1
    ):
        self.logger = setup_logger()
        self.rank = rank
        self.world_size = world_size
        
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
        
    def _gather_metrics(self, loss: float, samples: int) -> tuple[float, int]:
        """Gather metrics from all processes."""
        # Convert to tensors
        loss_tensor = torch.tensor([loss])
        samples_tensor = torch.tensor([samples])
        
        # Gather from all processes
        all_losses = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
        all_samples = [torch.zeros_like(samples_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(all_losses, loss_tensor)
        dist.all_gather(all_samples, samples_tensor)
        
        # Sum across processes
        total_loss = sum(l.item() for l in all_losses)
        total_samples = sum(s.item() for s in all_samples)
        
        return total_loss, total_samples
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=self.rank != 0) as pbar:
            for batch in pbar:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                batch_size = batch.y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                if self.rank == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Gather metrics from all processes
        total_loss, total_samples = self._gather_metrics(total_loss, total_samples)
        
        return total_loss / total_samples
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)
                batch_size = batch.y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Gather metrics from all processes
        total_loss, total_samples = self._gather_metrics(total_loss, total_samples)
        
        return total_loss / total_samples
    
    def test(self) -> Dict[str, float]:
        if not self.test_loader:
            raise ValueError("Test loader not provided")
            
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                outputs = self.model(batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        return calculate_metrics(all_labels, all_preds)
    
    def train(self, num_epochs: int, save_path: str = 'checkpoints'):
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Log progress
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}'
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
            metrics = self.test()
            for name, value in metrics.items():
                self.logger.info(f'Test {name}: {value:.4f}') 
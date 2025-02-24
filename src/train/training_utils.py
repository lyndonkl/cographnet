import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class EarlyStopping:
    """Early stopping utility."""
    patience: int = 7
    min_delta: float = 0.0
    counter: int = 0
    best_loss: Optional[float] = None
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class ModelCheckpoint:
    """Model checkpointing utility."""
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, model: torch.nn.Module, metrics: Dict[str, Any]) -> None:
        current = metrics[self.monitor]
        if ((self.mode == 'min' and current < self.best_value) or
            (self.mode == 'max' and current > self.best_value)):
            self.best_value = current
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, self.filepath) 
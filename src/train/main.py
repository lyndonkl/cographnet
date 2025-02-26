import os
import argparse
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import warnings
import urllib3
import torch.utils.data

from ..models import CoGraphNet
from ..data.document_dataset import create_dataloaders
from .trainer import CoGraphTrainer
from .distributed import setup_distributed, cleanup_distributed
from .training_utils import EarlyStopping, ModelCheckpoint
from .utils import setup_logger

# Suppress specific warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning)  # Suppresses PyTorch UserWarnings

def train_distributed(rank: int, world_size: int, args):
    """Distributed training function."""
    logger = setup_logger()
    setup_distributed(rank, world_size)
    
    try:
        # Create dataloaders using our new create_dataloaders function
        train_loader, val_loader, test_loader, num_classes = create_dataloaders(
            root=args.processed_graphs_dir,
            train_dir=args.train_path,
            val_dir=args.val_path,
            test_dir=args.test_path,
            batch_size=args.batch_size,
            num_workers=4,
            world_size=world_size,
            rank=rank
        )
        
        # Create model
        model = CoGraphNet(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_classes=num_classes,
            num_word_layers=args.num_word_layers
        )
        
        model = DistributedDataParallel(model)
        
        # Setup training utilities
        early_stopping = EarlyStopping(patience=args.patience)
        checkpoint = ModelCheckpoint(
            os.path.join(args.save_dir, 'best_model.pt'),
            monitor='val_loss'
        )
        
        # Create trainer
        trainer = CoGraphTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            rank=rank,
            world_size=world_size,
            num_epochs=args.epochs
        )
        
        # Train
        for epoch in range(args.epochs):
            # Set epoch for distributed sampling
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            
            train_loss = trainer.train_epoch(epoch)
            torch.distributed.barrier()
            
            val_loss, val_acc = trainer.validate()
            torch.distributed.barrier()
            
            trainer.scheduler.step()
            torch.distributed.barrier()
            
            # Save checkpoints and log on rank 0
            if rank == 0:
                metrics = {'val_loss': val_loss, 'val_acc': val_acc}
                checkpoint(model, metrics)
                
                logger.info(
                    f'Epoch {epoch}: '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Val Acc: {val_acc:.4f}'
                )
            
            # Synchronize early stopping across processes
            should_stop = torch.tensor([early_stopping(val_loss)], dtype=torch.bool)
            torch.distributed.broadcast(should_stop, src=0)
            
            if should_stop.item():
                if rank == 0:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        torch.distributed.barrier()
        
        # Test on all processes and gather results
        test_metrics = trainer.test()
        
        # Log results only on rank 0
        if rank == 0:
            logger.info(f"Test metrics: {test_metrics}")
            
    except Exception as e:
        logger.error(f"Rank {rank} failed with error: {str(e)}")
        raise e
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                      help='Path to processed training documents')
    parser.add_argument('--val_path', type=str, required=True,
                      help='Path to processed validation documents')
    parser.add_argument('--test_path', type=str, required=True,
                      help='Path to processed test documents')
    parser.add_argument('--processed_graphs_dir', type=str, default='processed_graphs',
                      help='Directory to store processed graph data')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--num_word_layers', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=7)
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.processed_graphs_dir, exist_ok=True)
    
    # Use CPU cores for distributed training
    world_size = mp.cpu_count()
    
    # Run distributed training
    mp.spawn(
        train_distributed,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main() 
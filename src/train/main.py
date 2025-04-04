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
import torch.distributed as dist


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning)
mp.set_start_method('spawn', force=True)

def compute_class_weights(train_loader, num_classes, device, rank, world_size):
    """Compute class weights across all ranks in DistributedDataParallel (DDP), ensuring proper synchronization."""

    # Ensure all processes start at the same point
    dist.barrier()

    # Collect all labels locally
    local_labels = []
    for batch in train_loader:
        local_labels.append(batch.y.cpu().numpy())  # Store labels from the current rank

    # Flatten and convert to tensor
    local_labels = np.concatenate(local_labels, axis=0) if local_labels else np.array([], dtype=np.int64)
    local_labels_tensor = torch.tensor(local_labels, dtype=torch.long, device=device)

    # Allocate a list to store gathered tensors
    gathered_labels_list = [torch.zeros_like(local_labels_tensor) for _ in range(world_size)]

    # Synchronize and gather all labels from all ranks
    dist.barrier()
    dist.all_gather(gathered_labels_list, local_labels_tensor)

    # Convert gathered tensors to numpy arrays
    all_labels = torch.cat(gathered_labels_list, dim=0).cpu().numpy()
    # Ensure all ranks wait for gathering to complete
    dist.barrier()

    if rank == 0:
        # Compute class weights using all gathered labels
        unique_labels = np.unique(all_labels)
        computed_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_labels,
            y=all_labels
        )

        # Initialize full weight tensor with zeros
        full_class_weights = np.zeros(num_classes)
        
        # Assign computed weights to their respective indices
        for label, weight in zip(unique_labels, computed_weights):
            full_class_weights[label] = weight  

        # Convert to tensor and send to the correct device
        class_weights_tensor = torch.tensor(full_class_weights, dtype=torch.float, device=device)
    else:
        # Placeholder tensor for other ranks
        class_weights_tensor = torch.zeros(num_classes, dtype=torch.float, device=device)

    # Synchronize before broadcasting
    dist.barrier()

    # Broadcast the computed weights from rank 0 to all other ranks
    dist.broadcast(class_weights_tensor, src=0)

    # Final synchronization to ensure all ranks have the same weights
    dist.barrier()

    return class_weights_tensor

def get_training_stage(epoch):
    if epoch < 25:
        return "sentence"
    elif epoch < 50:
        return "word"
    elif epoch < 75:
        return "fusion"
    else:
        return "fine_tune"

def reset_learning_rate(trainer, new_lr):
    """Manually reset learning rate when transitioning to a new training stage."""
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = new_lr
    trainer.logger.info(f"Learning rate reset to {new_lr} for new training stage.")


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
            rank=rank,
            n_splits=5,
            current_fold=rank % 5
        )
        
        # Create model
        model = CoGraphNet(
            word_in_channels=args.input_dim,
            sent_in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_word_layers,
            num_classes=num_classes
        )

        # Determine the device for training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the correct device
        model.to(device)
        
        model = DistributedDataParallel(model, find_unused_parameters=True)
        
        # Setup training utilities
        checkpoint = ModelCheckpoint(
            os.path.join(args.save_dir, 'best_model.pt'),
            monitor='val_loss'
        )

        patience_per_stage = {"sentence": 10, "word": 10, "fusion": 5, "fine_tune": 5}
        early_stopping = {stage: EarlyStopping(patience=patience) for stage, patience in patience_per_stage.items()} 
        # Compute class weights only on rank 0
        class_weights = compute_class_weights(
            train_loader,
            num_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            rank=rank,
            world_size=world_size
        )

        # Broadcast class weights to all ranks
        if world_size > 1:
            dist.broadcast(class_weights, src=0)
            torch.distributed.barrier()  # Sync all ranks before continuing

        
        # Create trainer
        trainer = CoGraphTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            rank=rank,
            world_size=world_size,
            num_epochs=args.epochs,
            class_weights=class_weights
        )
        
        # Train
        epoch = 0
        while epoch < args.epochs:
            stage = get_training_stage(epoch)
            
            if epoch == 0:
                trainer.freeze_all_except_sentence()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=1e-4)
            elif epoch == 25:
                trainer.freeze_all_except_word()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=1e-4)
            elif epoch == 50:
                trainer.freeze_all_except_fusion()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=5e-3)
            elif epoch == 75:
                trainer.unfreeze_all()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=5e-3)

            # Set epoch for distributed sampling
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            
            train_loss = trainer.train_epoch(epoch)
            torch.distributed.barrier()
            
            val_loss, val_acc = trainer.validate()
            torch.distributed.barrier()
            
            if stage in ["fusion", "fine_tune"]:
                trainer.scheduler.step(val_loss)
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
            should_stop = torch.tensor([early_stopping[stage](val_loss)], dtype=torch.bool)
            torch.distributed.broadcast(should_stop, src=0)
            
            if should_stop.item():
                if stage == "fine_tune":
                    if rank == 0:
                        logger.info(f"Early stopping triggered at epoch {epoch}. Ending training.")
                    break
                else:
                    next_stage_start_epoch = 25 if stage == "sentence" else 50 if stage == "word" else 75
                    if rank == 0:
                        logger.info(f"Early stopping triggered for stage {stage} at epoch {epoch}. Moving to next stage {get_training_stage(next_stage_start_epoch)}.")
                    epoch = next_stage_start_epoch  # Move to next stage start
                    continue
        
            epoch += 1
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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--num_word_layers', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=135)
    parser.add_argument('--patience', type=int, default=20)
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
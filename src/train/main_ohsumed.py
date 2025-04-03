import os
import argparse
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import warnings
import urllib3
import torch.utils.data
from collections import defaultdict

from ..models import CoGraphNet
from ..data.document_dataset_ohsumed import create_dataloaders_ohsumed
from .trainer import CoGraphTrainer
from .distributed import setup_distributed, cleanup_distributed
from .training_utils import EarlyStopping, ModelCheckpoint
from .utils import setup_logger, plot_overall_metrics
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
    if epoch < 100:
        return "sentence"
    elif epoch < 200:
        return "word"
    elif epoch < 300:
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
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        # Create dataloaders using our OHSUMED create_dataloaders function
        train_loader, val_loader, test_loader, num_classes = create_dataloaders_ohsumed(
            root=args.processed_graphs_dir,
            train_dir=args.train_path,
            test_dir=args.test_path,
            batch_size=args.batch_size,
            num_workers=4,
            world_size=world_size,
            rank=rank,
            val_split=0.2  # 80-20 split for train-val
        )

        # Optimized dropout configuration
        dropout_rate = {
            'word': 0.3743377153261378,
            'sent': 0.3934874479157007,
            'fusion': 0.237207820271752,
            'co_graph': 0.20629930383280137,
            'final': 0.239952468532074
        }

        dropout_config = {
            'word': True,      # dropout_word_enabled = 1
            'sent': False,     # dropout_sent_enabled = 0
            'fusion': True,    # dropout_fusion_enabled = 1
            'co_graph': False, # dropout_co_graph_enabled = 0
            'final': False     # dropout_final_enabled = 0
        }
        
        # Create model with optimized parameters
        model = CoGraphNet(
            word_in_channels=args.input_dim,
            sent_in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            num_word_layers=args.num_word_layers,
            num_sent_layers=args.num_sent_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            dropout_config=dropout_config
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

        patience_per_stage = {"sentence": 20, "word": 20, "fusion": 20, "fine_tune": 20}
        early_stopping = {stage: EarlyStopping(patience=patience) for stage, patience in patience_per_stage.items()} 
        # Compute class weights only on rank 0
        train_class_weights = compute_class_weights(
            train_loader,
            num_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            rank=rank,
            world_size=world_size
        )
        print(f"Train class weights: {train_class_weights}")

        val_class_weights = compute_class_weights(
            val_loader,
            num_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            rank=rank,
            world_size=world_size
        )

        print(f"Val class weights: {val_class_weights}")

        test_class_weights = compute_class_weights(
            test_loader,
            num_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            rank=rank,
            world_size=world_size
        )

        print(f"Test class weights: {test_class_weights}")

        # Broadcast class weights to all ranks
        if world_size > 1:
            dist.broadcast(train_class_weights, src=0)
            torch.distributed.barrier()  # Sync all ranks before continuing

        
        # Create trainer with optimized parameters
        trainer = CoGraphTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            rank=rank,
            world_size=world_size,
            num_epochs=args.epochs,
            train_class_weights=train_class_weights,
            val_class_weights=val_class_weights,
            test_class_weights=test_class_weights,
            gamma=args.gamma,
            plot_dir=os.path.join(args.save_dir, 'plots'),
            num_classes=num_classes
        )
        
        # Track metrics across phases
        overall_metrics = defaultdict(list)
        
        # Train
        epoch = 0
        while epoch < args.epochs:
            stage = get_training_stage(epoch)
            
            if epoch == 0:
                trainer.freeze_all_except_sentence()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=1e-4)
            elif epoch == 100:
                trainer.freeze_all_except_word()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=1e-4)
            elif epoch == 200:
                trainer.freeze_all_except_fusion()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=5e-3)
            elif epoch == 300:
                trainer.unfreeze_all()
                trainer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, trainer.model.parameters()), lr=5e-3)

            # Set epoch for distributed sampling
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            
            train_loss = trainer.train_epoch(epoch, stage)
            torch.distributed.barrier()
            
            val_loss, val_acc = trainer.validate(epoch, stage)
            torch.distributed.barrier()
            
            if stage in ["fusion", "fine_tune"]:
                trainer.scheduler.step(val_loss)
                torch.distributed.barrier()
            
            # Save checkpoints and log on rank 0
            if rank == 0:
                metrics = {'val_loss': val_loss, 'val_acc': val_acc}
                checkpoint(model, metrics)
                
                # Track metrics for plotting
                overall_metrics['train_loss'].append(train_loss)
                overall_metrics['val_loss'].append(val_loss)
                overall_metrics['val_acc'].append(val_acc)
                
                # Plot overall metrics
                plot_overall_metrics(
                    overall_metrics,
                    stage,
                    epoch,
                    os.path.join(args.save_dir, 'plots')
                )
                
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
                    next_stage_start_epoch = 100 if stage == "sentence" else 200 if stage == "word" else 300
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
    parser.add_argument('--test_path', type=str, required=True,
                      help='Path to processed test documents')
    parser.add_argument('--processed_graphs_dir', type=str, default='processed_graphs_ohsumed',
                      help='Directory to store processed graph data')
    parser.add_argument('--save_dir', type=str, default='checkpoints_ohsumed')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=111)
    parser.add_argument('--num_word_layers', type=int, default=1)
    parser.add_argument('--num_sent_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0009841767327191644)
    parser.add_argument('--weight_decay', type=float, default=2.646938537392353e-08)
    parser.add_argument('--gamma', type=float, default=3.1139542593286)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.processed_graphs_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    
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
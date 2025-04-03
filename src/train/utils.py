import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import torch

def setup_logger() -> logging.Logger:
    """Setup and return a logger instance."""
    logger = logging.getLogger('cograph')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    
    ece = 0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_probs = probs[mask]
            bin_labels = labels[mask]
            bin_acc = np.mean(bin_labels == np.argmax(bin_probs, axis=1))
            bin_conf = np.mean(np.max(bin_probs, axis=1))
            bin_size = np.sum(mask)
            ece += np.abs(bin_acc - bin_conf) * (bin_size / len(labels))
    
    return ece

def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """Plot reliability diagram for model calibration."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_probs = probs[mask]
            bin_labels = labels[mask]
            bin_acc = np.mean(bin_labels == np.argmax(bin_probs, axis=1))
            bin_conf = np.mean(np.max(bin_probs, axis=1))
            bin_size = np.sum(mask)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_sizes.append(bin_size)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confs, bin_accs, 'o-', label='Model Calibration')
    plt.fill_between(bin_confs, bin_accs, bin_confs, alpha=0.2)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_prediction_distribution(
    predictions: List[int],
    class_map: Dict[str, int],
    phase: int,
    epoch: int,
    save_dir: str
) -> None:
    """Plot histogram of predicted classes."""
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, bins=len(class_map))
    plt.title(f'Prediction Distribution (Phase {phase}, Epoch {epoch+1})')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'pred_dist_phase_{phase}_epoch_{epoch+1}.png'))
    plt.close()

def plot_class_distribution(
    predictions: List[int],
    labels: List[int],
    class_map: Dict[str, int],
    phase: int,
    epoch: int,
    save_dir: str
) -> None:
    """Plot bar chart comparing predicted vs true class counts."""
    pred_counts = np.bincount(predictions, minlength=len(class_map))
    true_counts = np.bincount(labels, minlength=len(class_map))
    
    x = np.arange(len(class_map))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
    plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
    plt.title(f'Class Distribution (Phase {phase}, Epoch {epoch+1})')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'class_dist_phase_{phase}_epoch_{epoch+1}.png'))
    plt.close()

def plot_unique_classes_per_batch(
    batch_indices: List[int],
    unique_pred: List[int],
    unique_true: List[int],
    num_classes: int,
    phase: int,
    epoch: int,
    save_dir: str
) -> None:
    """Plot unique classes per batch during training."""
    plt.figure(figsize=(12, 6))
    plt.plot(batch_indices, unique_pred, label='Unique Predictions', alpha=0.7)
    plt.plot(batch_indices, unique_true, label='Unique True Labels', alpha=0.7)
    plt.axhline(y=num_classes, color='r', linestyle='--', label='Total Classes')
    plt.title(f'Unique Classes per Batch (Phase {phase}, Epoch {epoch+1})')
    plt.xlabel('Batch Index')
    plt.ylabel('Number of Unique Classes')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'unique_classes_phase_{phase}_epoch_{epoch+1}.png'))
    plt.close()

def plot_overall_metrics(
    metrics: Dict[str, List[float]],
    phase: int,
    epoch: int,
    save_dir: str
) -> None:
    """Plot overall training and validation metrics across phases."""
    plt.figure(figsize=(12, 6))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name, alpha=0.7)
    plt.title(f'Overall Metrics (Phase {phase}, Epoch {epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'overall_metrics_phase_{phase}_epoch_{epoch+1}.png'))
    plt.close() 
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
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    max_probs = probs.max(axis=1)
    pred_labels = probs.argmax(axis=1)
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == pred_labels[in_bin])
            avg_confidence_in_bin = np.mean(max_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """Plot reliability diagram for model calibration."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    max_probs = probs.max(axis=1)
    pred_labels = probs.argmax(axis=1)
    accuracies = []
    confidences = []
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        if np.sum(in_bin) > 0:
            accuracy = np.mean(labels[in_bin] == pred_labels[in_bin])
            confidence = np.mean(max_probs[in_bin])
        else:
            accuracy = 0.0
            confidence = 0.0
        accuracies.append(accuracy)
        confidences.append(confidence)
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, accuracies, marker='o', label="Empirical Accuracy")
    plt.plot(bin_centers, confidences, marker='s', label="Average Confidence")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

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
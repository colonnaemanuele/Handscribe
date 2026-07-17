import torch
import numpy as np
import os
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class MetricMode(Enum):
    """Direction for metric optimization"""
    MINIMIZE = "min"  # Lower is better (e.g., loss, WER)
    MAXIMIZE = "max"  # Higher is better (e.g., accuracy, BLEU)


@dataclass
class MetricConfig:
    """Configuration for a metric to track"""
    name: str
    mode: MetricMode
    weight: float = 1.0  # Weight for combined metrics
    

@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint"""
    epoch: int
    metric_value: float
    metric_name: str
    filepath: str
    metrics_dict: Dict[str, float] = field(default_factory=dict)


class CheckpointManager:
    """
    Manages model checkpoints based on multiple metrics.
    
    Features:
    - Track best models for different metrics (dev/test BLEU, ROUGE, combined)
    - Automatic cleanup of old checkpoints
    - Support for combined metrics with custom weights
    """
    
    def __init__(
        self, 
        work_dir: str,
        keep_best_n: int = 3,
        keep_periodic: int = 2,
        recoder = None
    ):
        """
        Args:
            work_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep per metric
            keep_periodic: Number of periodic checkpoints to keep
            recoder: Logger object for printing messages
        """
        self.work_dir = work_dir
        self.keep_best_n = keep_best_n
        self.keep_periodic = keep_periodic
        self.recoder = recoder
        
        # Track best checkpoints for each metric
        self.best_checkpoints: Dict[str, List[CheckpointInfo]] = {}
        self.periodic_checkpoints: List[str] = []
        
        # Track overall best values
        self.best_values: Dict[str, float] = {}
        self.best_epochs: Dict[str, int] = {}
        
    def _log(self, message: str):
        """Helper to log messages"""
        if self.recoder:
            self.recoder.print_log(message)
        else:
            print(message)
    
    def compute_combined_metric(
        self, 
        metrics: Dict[str, float], 
        metric_configs: List[MetricConfig]
    ) -> float:
        """
        Compute a weighted combination of multiple metrics.
        
        Args:
            metrics: Dictionary of metric values
            metric_configs: List of metrics to combine with their weights
            
        Returns:
            Combined metric value (normalized)
        """
        total_weight = sum(config.weight for config in metric_configs)
        combined = 0.0
        
        for config in metric_configs:
            if config.name in metrics:
                value = metrics[config.name]
                # Normalize to 0-100 scale if needed
                if value < 1.0:
                    value *= 100
                combined += value * config.weight
                
        return combined / total_weight if total_weight > 0 else 0.0
    
    def is_better(
        self, 
        new_value: float, 
        old_value: float, 
        mode: MetricMode
    ) -> bool:
        """Check if new value is better than old value"""
        if mode == MetricMode.MAXIMIZE:
            return new_value > old_value
        else:
            return new_value < old_value
    
    def should_save(
        self, 
        metric_name: str,
        metric_value: float,
        mode: MetricMode
    ) -> bool:
        """
        Check if current metric warrants saving a checkpoint.
        
        Args:
            metric_name: Name of the metric
            metric_value: Current value of the metric
            mode: Whether to minimize or maximize the metric
            
        Returns:
            True if checkpoint should be saved
        """
        if metric_name not in self.best_values:
            return True
            
        return self.is_better(metric_value, self.best_values[metric_name], mode)
    
    def update_and_save(
        self,
        epoch: int,
        model,
        optimizer,
        scaler,
        rng,
        metrics: Dict[str, float],
        metric_name: str,
        mode: MetricMode,
        checkpoint_tag: str = ""
    ) -> Optional[str]:
        """
        Update tracking and save checkpoint if it's one of the best.
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            scaler: Gradient scaler to save
            rng: Random number generator state
            metrics: Dictionary of all metrics for this epoch
            metric_name: Primary metric to use for this checkpoint
            mode: Optimization direction for the metric
            checkpoint_tag: Optional tag to add to filename
            
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        metric_value = metrics.get(metric_name, float('inf'))
        
        # Check if this is a new best
        is_new_best = False
        if metric_name not in self.best_values:
            is_new_best = True
        else:
            is_new_best = self.is_better(metric_value, self.best_values[metric_name], mode)
        
        if is_new_best:
            self.best_values[metric_name] = metric_value
            self.best_epochs[metric_name] = epoch
            
            # Create checkpoint filename
            tag_str = f"_{checkpoint_tag}" if checkpoint_tag else ""
            filename = f"best_{metric_name.replace('/', '_')}_epoch{epoch}{tag_str}_{metric_value:.4f}.pt"
            filepath = os.path.join(self.work_dir, filename)
            
            # Save checkpoint
            self._save_checkpoint(filepath, epoch, model, optimizer, scaler, rng)
            
            # Track this checkpoint
            checkpoint_info = CheckpointInfo(
                epoch=epoch,
                metric_value=metric_value,
                metric_name=metric_name,
                filepath=filepath,
                metrics_dict=metrics.copy()
            )
            
            if metric_name not in self.best_checkpoints:
                self.best_checkpoints[metric_name] = []
            
            self.best_checkpoints[metric_name].append(checkpoint_info)
            self.best_checkpoints[metric_name].sort(
                key=lambda x: x.metric_value,
                reverse=(mode == MetricMode.MAXIMIZE)
            )
            
            # Clean up old checkpoints for this metric
            self._cleanup_metric_checkpoints(metric_name)
            
            self._log(f"✓ New best {metric_name}: {metric_value:.4f} at epoch {epoch}")
            self._log(f"  Saved: {filename}")
            
            return filepath
        
        return None
    
    def save_periodic(
        self,
        epoch: int,
        model,
        optimizer,
        scaler,
        rng,
        metrics: Dict[str, float]
    ) -> str:
        """
        Save a periodic checkpoint.
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            scaler: Gradient scaler to save
            rng: Random number generator state
            metrics: Dictionary of metrics for logging
            
        Returns:
            Path to saved checkpoint
        """
        filename = f"periodic_epoch{epoch}.pt"
        filepath = os.path.join(self.work_dir, filename)
        
        self._save_checkpoint(filepath, epoch, model, optimizer, scaler, rng)
        self.periodic_checkpoints.append(filepath)
        
        # Clean up old periodic checkpoints
        if len(self.periodic_checkpoints) > self.keep_periodic:
            old_checkpoints = self.periodic_checkpoints[:-self.keep_periodic]
            for old_path in old_checkpoints:
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        self._log(f"  Removed old periodic checkpoint: {os.path.basename(old_path)}")
                    except OSError as e:
                        self._log(f"  Warning: Could not remove {old_path}: {e}")
            self.periodic_checkpoints = self.periodic_checkpoints[-self.keep_periodic:]
        
        self._log(f"Saved periodic checkpoint: {filename}")
        return filepath
    
    def _save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        model,
        optimizer,
        scaler,
        rng
    ):
        """Internal method to save checkpoint to disk"""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": optimizer.scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": rng.save_rng_state(),
            },
            filepath,
        )
    
    def _cleanup_metric_checkpoints(self, metric_name: str):
        """Remove checkpoints beyond keep_best_n for a specific metric"""
        if metric_name not in self.best_checkpoints:
            return
        
        checkpoints = self.best_checkpoints[metric_name]
        if len(checkpoints) > self.keep_best_n:
            to_remove = checkpoints[self.keep_best_n:]
            for checkpoint_info in to_remove:
                if os.path.exists(checkpoint_info.filepath):
                    try:
                        os.remove(checkpoint_info.filepath)
                        self._log(f"  Removed old checkpoint: {os.path.basename(checkpoint_info.filepath)}")
                    except OSError as e:
                        self._log(f"  Warning: Could not remove {checkpoint_info.filepath}: {e}")
            
            self.best_checkpoints[metric_name] = checkpoints[:self.keep_best_n]
    
    def get_best_checkpoint(self, metric_name: str) -> Optional[CheckpointInfo]:
        """Get the best checkpoint for a specific metric"""
        if metric_name in self.best_checkpoints and self.best_checkpoints[metric_name]:
            return self.best_checkpoints[metric_name][0]
        return None
    
    def print_summary(self):
        """Print summary of all tracked metrics and their best values"""
        self._log("\n" + "="*80)
        self._log("Best Metrics Summary:")
        self._log("="*80)
        
        for metric_name in sorted(self.best_values.keys()):
            best_value = self.best_values[metric_name]
            best_epoch = self.best_epochs[metric_name]
            self._log(f"  {metric_name:30s}: {best_value:8.4f} (epoch {best_epoch})")
        
        self._log("="*80 + "\n")

def _is_loss_valid(loss):
    """Check if loss is finite and valid"""
    if loss is None:
        return False
    try:
        if isinstance(loss, torch.Tensor):
            return torch.isfinite(loss).all().item()
        else:
            return np.isfinite(loss)
    except Exception:
        try:
            loss_cpu = loss.detach().cpu().numpy()
            return np.isfinite(loss_cpu).all()
        except Exception:
            return False
        
def _log_nan_debug_info(batch_idx, epoch_idx, ret_dict, recoder):
    """Log debugging information when NaN loss is encountered"""
    recoder.print_log(f'NaN/Inf loss detected at batch {batch_idx}, epoch {epoch_idx}')
    
    if isinstance(ret_dict, dict):
        recoder.print_log(f"ret_dict keys: {list(ret_dict.keys())}")
        for k, v in ret_dict.items():
            try:
                if isinstance(v, torch.Tensor):
                    v_cpu = v.detach().cpu()
                    recoder.print_log(
                        f"{k}: shape={v_cpu.shape}, dtype={v_cpu.dtype}, "
                        f"min={v_cpu.min().item():.6f}, max={v_cpu.max().item():.6f}, "
                        f"mean={v_cpu.mean().item():.6f}, "
                        f"nan={torch.isnan(v_cpu).any().item()}, "
                        f"inf={torch.isinf(v_cpu).any().item()}"
                    )
                else:
                    recoder.print_log(f"{k}: type={type(v)}")
            except Exception as e:
                recoder.print_log(f"Could not inspect ret_dict['{k}']: {e}")
    else:
        recoder.print_log(f"ret_dict is not a dict: {type(ret_dict)}")

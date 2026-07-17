import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction="batchmean")
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:] / self.T, dim=-1).view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:] / self.T, dim=-1).view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs) * self.T * self.T
        # mask_probs = F.softmax(ref_logits[:, :, 1:], dim=-1).view(-1, ref_logits.shape[2] - 1)
        # mask = torch.max(mask_probs, dim=1)[0] > 0.5
        # if torch.sum(mask) != 0:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask) / torch.sum(mask)
        # else:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask)
        return loss
    
class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean', label_smoothing=0.1, ignore_index=-100):
        super(CrossEntropy, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, 
            reduction=reduction,
            ignore_index=ignore_index  # Use -100 as standard padding ignore value
        )
        self.ignore_index = ignore_index

    def forward(self, prediction_output, target_labels):
        """
        Handle MBart outputs properly, accommodating mismatched sequence lengths.
        """
        if hasattr(prediction_output, 'loss') and prediction_output.loss is not None:
            return prediction_output.loss
        
        logits = prediction_output.logits if hasattr(prediction_output, 'logits') else prediction_output
        batch_size, seq_len, num_classes = logits.shape
        
        # Handle case where target_labels might be passed as a scalar (wrong usage)
        if not isinstance(target_labels, torch.Tensor):
            raise TypeError(f"target_labels must be a torch.Tensor, got {type(target_labels)}")
        
        device = logits.device
        target_labels = target_labels.to(device)
        
        # Reshape logits to [batch_size * seq_len, num_classes]
        reshaped_logits = logits.contiguous().view(-1, num_classes)
        
        # Handle different target label shapes
        if target_labels.dim() == 2:
            # Case: target_labels has shape [batch_size, target_seq_len]
            target_seq_len = target_labels.size(1)
            
            if target_seq_len > seq_len:
                # Truncate target to match prediction length
                reshaped_labels = target_labels[:, :seq_len].contiguous().view(-1)
            elif target_seq_len < seq_len:
                # Pad target with ignore_index
                padded_labels = torch.full(
                    (batch_size, seq_len), 
                    self.ignore_index,
                    dtype=target_labels.dtype, 
                    device=device
                )
                padded_labels[:, :target_seq_len] = target_labels
                reshaped_labels = padded_labels.contiguous().view(-1)
            else:
                # Same length, just reshape
                reshaped_labels = target_labels.contiguous().view(-1)
                
        elif target_labels.dim() == 1:
            # Case: target_labels already flattened
            target_len = target_labels.size(0)
            expected_len = batch_size * seq_len
            
            if target_len > expected_len:
                reshaped_labels = target_labels[:expected_len]
            elif target_len < expected_len:
                padded_labels = torch.full(
                    (expected_len,), 
                    self.ignore_index, 
                    dtype=target_labels.dtype, 
                    device=device
                )
                padded_labels[:target_len] = target_labels
                reshaped_labels = padded_labels
            else:
                reshaped_labels = target_labels
        else:
            raise ValueError(f"Unexpected target_labels dimension: {target_labels.dim()}")
        
        # Validate label values
        valid_mask = (reshaped_labels >= 0) & (reshaped_labels < num_classes)
        invalid_mask = ~valid_mask & (reshaped_labels != self.ignore_index)
        
        if invalid_mask.any():
            reshaped_labels = reshaped_labels.clone()
            reshaped_labels[invalid_mask] = self.ignore_index
        
        return self.loss_fn(reshaped_logits, reshaped_labels)


class CrossRougeEntropy(nn.Module):
    """
    CrossEntropy loss with adaptive weighting based on prediction quality.
    
    This loss increases the penalty for samples where the model makes more errors,
    encouraging the model to focus on harder examples.
    
    Key improvements:
    1. Uses prediction confidence (entropy) instead of ROUGE to weight samples
    2. Properly handles padding with ignore_index
    3. More stable gradient flow
    """

    def __init__(self, rouge_weight=1.0, label_smoothing=0.1, ignore_index=-100, 
                 focal_alpha=0.25, focal_gamma=2.0, use_focal=False):
        super(CrossRougeEntropy, self).__init__()
        self.rouge_weight = rouge_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Base loss function without reduction (we'll reduce manually)
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none',
            ignore_index=ignore_index
        )

    def compute_token_accuracy(self, predictions, targets):
        """
        Compute per-sample token-level accuracy.
        
        Args:
            predictions: Predicted token indices [batch_size, seq_len]
            targets: Target token indices [batch_size, seq_len]
            
        Returns:
            Accuracy scores per batch [batch_size], range [0, 1]
        """
        batch_size = predictions.size(0)
        accuracies = torch.zeros(batch_size, device=targets.device)
        
        for i in range(batch_size):
            # Only consider non-padding tokens
            valid_mask = targets[i] != self.ignore_index
            if valid_mask.sum() == 0:
                accuracies[i] = 1.0  # No valid tokens, no penalty
                continue
                
            pred_valid = predictions[i][valid_mask]
            target_valid = targets[i][valid_mask]
            
            # Calculate accuracy
            matches = (pred_valid == target_valid).float().sum()
            total = valid_mask.float().sum()
            accuracies[i] = matches / total if total > 0 else 1.0
                    
        return accuracies

    def compute_prediction_confidence(self, logits, targets):
        """
        Compute average prediction confidence for each sample.
        Higher confidence = model is more certain about predictions.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target indices [batch_size, seq_len]
            
        Returns:
            Confidence scores per batch [batch_size], range [0, 1]
        """
        batch_size, seq_len, vocab_size = logits.shape
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of predicted class
        pred_probs = probs.gather(2, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)
        
        confidences = torch.zeros(batch_size, device=targets.device)
        
        for i in range(batch_size):
            valid_mask = targets[i] != self.ignore_index
            if valid_mask.sum() == 0:
                confidences[i] = 1.0
                continue
                
            # Average confidence over valid tokens
            confidences[i] = pred_probs[i][valid_mask].mean()
        
        return confidences

    def _prepare_targets(self, target_labels, batch_size, seq_len, device):
        """
        Prepare target labels to match [batch_size, seq_len] format.
        Pads with ignore_index instead of 0.
        """
        if target_labels.dim() == 2:
            orig_batch, orig_seq = target_labels.size()
            
            if orig_batch == batch_size and orig_seq == seq_len:
                return target_labels
            
            # Create padded tensor
            padded = torch.full(
                (batch_size, seq_len),
                self.ignore_index,
                dtype=target_labels.dtype,
                device=device
            )
            
            # Copy valid data
            valid_batch = min(batch_size, orig_batch)
            valid_seq = min(seq_len, orig_seq)
            padded[:valid_batch, :valid_seq] = target_labels[:valid_batch, :valid_seq]
            
            return padded
            
        elif target_labels.dim() == 1:
            # Reshape 1D to 2D preserving batch structure
            padded = torch.full(
                (batch_size, seq_len),
                self.ignore_index,
                dtype=target_labels.dtype,
                device=device
            )
            
            total_elements = target_labels.size(0)
            
            for b in range(batch_size):
                start_idx = b * seq_len
                if start_idx >= total_elements:
                    break
                
                elements_to_copy = min(seq_len, total_elements - start_idx)
                padded[b, :elements_to_copy] = target_labels[start_idx:start_idx + elements_to_copy]
            
            return padded
        else:
            raise ValueError(f"Unexpected target dimension: {target_labels.dim()}")

    def forward(self, prediction_output, target_labels):
        """
        Compute weighted cross-entropy loss.
        
        The weight increases for samples where the model has:
        - Lower prediction confidence (uncertain)
        - Lower accuracy (making more mistakes)
        
        This creates an adaptive curriculum where harder samples get more attention.
        """
        if not isinstance(target_labels, torch.Tensor):
            raise TypeError(f"target_labels must be a torch.Tensor, got {type(target_labels)}")

        # Extract logits
        logits = prediction_output.logits if hasattr(prediction_output, 'logits') else prediction_output
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Prepare targets
        target_labels = target_labels.to(device)
        targets_2d = self._prepare_targets(target_labels, batch_size, seq_len, device)
        
        # Compute base loss (per-token)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets_2d.view(-1)
        
        token_losses = self.loss_fn(logits_flat, targets_flat)  # [batch_size * seq_len]
        
        # Reshape to [batch_size, seq_len]
        token_losses = token_losses.view(batch_size, seq_len)
        
        # Create mask for valid (non-padding) tokens
        valid_mask = (targets_2d != self.ignore_index).float()
        
        # Compute per-sample loss (average over valid tokens)
        sample_losses = (token_losses * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        
        if self.rouge_weight == 0:
            # No adaptive weighting, just return mean
            return sample_losses.mean()
        
        # Compute adaptive weights based on prediction quality
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            
            # Option 1: Use token accuracy (lower accuracy = higher weight)
            accuracies = self.compute_token_accuracy(predictions, targets_2d)
            
            # Option 2: Use prediction confidence (lower confidence = higher weight)
            # confidences = self.compute_prediction_confidence(logits, targets_2d)
            
            # Create weight factor: samples with lower accuracy get higher weight
            # weight_factor ranges from 1.0 to (1.0 + rouge_weight)
            weight_factor = 1.0 + self.rouge_weight * (1.0 - accuracies)
            
            # Optional: Clip weights to prevent extreme values
            weight_factor = torch.clamp(weight_factor, min=0.5, max=2.0)
        
        # Apply weights
        weighted_losses = sample_losses * weight_factor
        
        # Optional: Apply focal loss-style weighting
        if self.use_focal:
            # Focal loss: down-weight easy examples
            p = torch.exp(-sample_losses)  # Probability of correct prediction
            focal_weight = self.focal_alpha * (1 - p) ** self.focal_gamma
            weighted_losses = weighted_losses * focal_weight
        
        return weighted_losses.mean()


class SimplifiedRougeEntropy(CrossEntropy):
    """
    Simplified version: Just add a small penalty term based on n-gram overlap.
    This is more interpretable and stable than the full CrossRougeEntropy.
    """
    
    def __init__(self, overlap_weight=0.1, label_smoothing=0.1, ignore_index=-100):
        super().__init__(reduction='mean', label_smoothing=label_smoothing, ignore_index=ignore_index)
        self.overlap_weight = overlap_weight
    
    def compute_unigram_overlap(self, predictions, targets):
        """
        Compute unigram overlap (similar to ROUGE-1 recall).
        
        Returns:
            Overlap score per sample [batch_size], higher is better
        """
        batch_size = predictions.size(0)
        overlaps = torch.zeros(batch_size, device=targets.device)
        
        for i in range(batch_size):
            valid_mask = targets[i] != self.ignore_index
            if valid_mask.sum() == 0:
                overlaps[i] = 1.0
                continue
            
            pred_set = set(predictions[i][valid_mask].cpu().tolist())
            target_set = set(targets[i][valid_mask].cpu().tolist())
            
            if len(target_set) == 0:
                overlaps[i] = 1.0
            else:
                overlap_count = len(pred_set & target_set)
                overlaps[i] = overlap_count / len(target_set)
        
        return overlaps
    
    def forward(self, prediction_output, target_labels):
        # Compute base cross-entropy loss
        base_loss = super().forward(prediction_output, target_labels)
        
        if self.overlap_weight == 0:
            return base_loss
        
        # Extract info for overlap computation
        logits = prediction_output.logits if hasattr(prediction_output, 'logits') else prediction_output
        batch_size, seq_len, _ = logits.shape
        
        # Prepare targets
        device = logits.device
        target_labels = target_labels.to(device)
        
        if target_labels.dim() == 1:
            targets_2d = target_labels.view(batch_size, -1)
            if targets_2d.size(1) < seq_len:
                padded = torch.full((batch_size, seq_len), self.ignore_index, 
                                  dtype=target_labels.dtype, device=device)
                padded[:, :targets_2d.size(1)] = targets_2d
                targets_2d = padded
            elif targets_2d.size(1) > seq_len:
                targets_2d = targets_2d[:, :seq_len]
        else:
            targets_2d = target_labels
        
        # Compute overlap penalty
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            overlaps = self.compute_unigram_overlap(predictions, targets_2d)
            
            # Penalty increases when overlap is low
            overlap_penalty = self.overlap_weight * (1.0 - overlaps).mean()
        
        return base_loss + overlap_penalty
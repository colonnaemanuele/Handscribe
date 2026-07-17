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
    def __init__(self, reduction='mean', label_smoothing=0.2, ignore_index=-100):
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
        
        device = target_labels.device
        reshaped_logits = logits.contiguous().view(-1, num_classes).to(device)
        
        # Handle different target label shapes
        if target_labels.dim() == 2:
            # Case: target_labels has shape [batch_size, target_seq_len]
            target_seq_len = target_labels.size(1)
            
            if target_seq_len > seq_len:
                # If target is longer than prediction, truncate target
                reshaped_labels = target_labels[:, :seq_len].contiguous().view(-1)
            elif target_seq_len < seq_len:
                padded_labels = torch.full((batch_size, seq_len), self.ignore_index,dtype=target_labels.dtype, device=device)
                padded_labels[:, :target_seq_len] = target_labels
                reshaped_labels = padded_labels.contiguous().view(-1)
            else:
                # If same length, just reshape
                reshaped_labels = target_labels.contiguous().view(-1)
                
        elif target_labels.dim() == 1:
            # Case: target_labels has shape [batch_size*target_seq_len] or similar
            target_len = target_labels.size(0)
            expected_len = batch_size * seq_len
            
            if target_len > expected_len:
                # If target is longer than needed, truncate
                reshaped_labels = target_labels[:expected_len]
            elif target_len < expected_len:
                padded_labels = torch.full((expected_len,), self.ignore_index, dtype=target_labels.dtype, device=device)
                padded_labels[:target_len] = target_labels
                reshaped_labels = padded_labels
            else:
                reshaped_labels = target_labels
        else:
            raise ValueError(f"Unexpected target_labels dimension: {target_labels.dim()}")
        
        valid_mask = (reshaped_labels >= 0) & (reshaped_labels < num_classes)
        invalid_mask = ~valid_mask & (reshaped_labels != self.ignore_index)
        
        if invalid_mask.any():
            # print(f"Warning: Found {invalid_mask.sum()} out-of-bounds target values. Max target: {reshaped_labels.max()}, Vocab size: {num_classes}")
            # Set out-of-bounds values to ignore_index
            reshaped_labels = reshaped_labels.clone()
            reshaped_labels[invalid_mask] = self.ignore_index
        
        # print(f"Reshaped logits: {reshaped_logits.shape}, Reshaped labels: {reshaped_labels.shape}")
        return self.loss_fn(reshaped_logits, reshaped_labels)


class CrossRougeEntropy(CrossEntropy):
    """
    CrossEntropy loss with ROUGE-based weighting.
    """

    def __init__(self, rouge_weight=1.0, label_smoothing=0.2, ignore_index=-100):
        super(CrossRougeEntropy, self).__init__(reduction='none', label_smoothing=label_smoothing, ignore_index=ignore_index)
        self.rouge_weight = rouge_weight
        self.ignore_index = ignore_index

    def compute_rouge(self, predictions, targets):
        """
        Compute a simplified ROUGE-like score between predictions and targets.
        
        Args:
            predictions: Tensor of predicted token indices [batch_size, seq_len]
            targets: Tensor of target token indices [batch_size, seq_len]
            
        Returns:
            Tensor of rouge scores per batch item [batch_size]
        """
        batch_size = predictions.size(0)
        rouge_scores = torch.zeros(batch_size, device=targets.device)
        
        for i in range(batch_size):
            valid_mask = targets[i] != self.ignore_index
            if valid_mask.sum() == 0:
                rouge_scores[i] = 1.0  # Default to 1.0 (no penalty) for empty targets
                continue
                
            # Filter to only valid tokens
            pred_valid = predictions[i][valid_mask]
            target_valid = targets[i][valid_mask]
            matches = (pred_valid == target_valid).float().sum()
            total = valid_mask.float().sum()   
            # Calculate Rouge-1 Precision-like score (matches / total)
            rouge_scores[i] = matches / total if total > 0 else 1.0
                    
        return rouge_scores
    
    def _reshape_targets(self, target_labels, batch_size, seq_len):
        """
        Reshape target labels to match the expected dimensions [batch_size, seq_len].
        This preserves the batch structure and only pads with 0s as needed.
        
        Args:
            target_labels: Input labels tensor (either 1D or 2D)
            batch_size: Desired batch size
            seq_len: Desired sequence length
            
        Returns:
            Tensor of shape [batch_size, seq_len] with proper padding (0)
        """
        device = target_labels.device
        padded_targets = torch.zeros((batch_size, seq_len), dtype=target_labels.dtype, device=device)
        
        # Handle 1D tensor case
        if target_labels.dim() == 1:
            # Reshape properly to preserve batch structure
            total_elements = target_labels.size(0)
            
            for b in range(batch_size):
                # Calculate start position for this batch
                start_idx = b * seq_len
                if start_idx >= total_elements:
                    break
                
                # Calculate how many elements we can copy
                elements_to_copy = min(seq_len, total_elements - start_idx)
                
                # Copy elements to the right position
                padded_targets[b, :elements_to_copy] = target_labels[start_idx:start_idx + elements_to_copy]
        
        # Handle 2D tensor case
        elif target_labels.dim() == 2:
            orig_batch, orig_seq = target_labels.size()
            
            # Copy as much as we can
            valid_batch = min(batch_size, orig_batch)
            valid_seq = min(seq_len, orig_seq)
            
            padded_targets[:valid_batch, :valid_seq] = target_labels[:valid_batch, :valid_seq]
        
        return padded_targets

    def forward(self, prediction_output, target_labels):
        if not isinstance(target_labels, torch.Tensor):
            raise TypeError(f"target_labels must be a torch.Tensor, got {type(target_labels)}")

        base_loss = super().forward(prediction_output, target_labels)
        logits = prediction_output.logits if hasattr(prediction_output, 'logits') else prediction_output
        device = target_labels.device
        logits = logits.to(device)
        
        batch_size, seq_len, _ = logits.shape
        
        if base_loss.dim() == 0:
            return base_loss #scalar 
        elif base_loss.dim() == 1:
            # If base_loss is 1D with shape [batch_size * seq_len]
            if base_loss.numel() == batch_size * seq_len:
                base_loss = base_loss.view(batch_size, seq_len)
            else:
                return base_loss.mean() # Unexpected dimensions, return mean
        elif base_loss.dim() == 2:
            # Already has correct shape [batch_size, seq_len]
            if base_loss.shape != (batch_size, seq_len):
                return base_loss.mean()
        else:
            return base_loss.mean() # Unexpected dimensions, return mean

        per_sample_loss = base_loss.sum(dim=1)
        
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            if target_labels.dim() == 1:
                target_labels = self._reshape_targets(target_labels, batch_size, seq_len)
            elif target_labels.size(1) != seq_len:
                target_labels = self._reshape_targets(target_labels, batch_size, seq_len)

            # Calculate rouge scores and weight factors
            rouge_scores = self.compute_rouge(predictions, target_labels)
            weight_factor = 1.0 + self.rouge_weight * (1.0 - rouge_scores)
        
        # Apply weights to per-sample losses - both should have shape [batch_size]
        if per_sample_loss.shape != weight_factor.shape:
            print(f"Warning: Shape mismatch in CrossRougeEntropy - per_sample_loss: {per_sample_loss.shape}, weight_factor: {weight_factor.shape}")
            return per_sample_loss.mean()
            
        weighted_loss = (per_sample_loss * weight_factor).mean()
        return weighted_loss
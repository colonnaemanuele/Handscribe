import os
import torch
import numpy as np
from tqdm import tqdm
import gc
import wandb
from handscribe.utils.helper import _is_loss_valid, _log_nan_debug_info
from handscribe.evaluation.slr_eval.rouge_blue_calculation import compute_rouge_bleu_batch
from torch.amp.autocast_mode import autocast as autocast

def seq_train(loader, model, optimizer, device, epoch_idx, recoder, scaler):
    model.train()
    loss_value = []
    
    accumulation_steps = 2
    
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    tqdm_loader = tqdm(loader, ncols=100)
    nan_count = 0
    
    for batch_idx, data in enumerate(tqdm_loader):
        vid = device.data_to_device(data[0])                # frame folder
        vid_lgt = device.data_to_device(data[1])            # frame length
        label = data[2]                                     # tokenized sentence
        label_lgt = data[3]                                 # gloss
        gt_sentences = [s.split('|')[-1] for s in data[4]]  # sentences    
        
        optimizer.zero_grad()
        with autocast(device_type=device.device_type):
            ret_dict = model(vid, vid_lgt, gt_sentences=gt_sentences)
            loss = model.criterion_calculation(ret_dict, label, label_lgt, gt_sentences)

        wandb.log({
            "predicted_sents": ret_dict.get('recognized_sents', []),
            "gt_sents": gt_sentences
        })
        
        if not _is_loss_valid(loss):
            nan_count += 1
            _log_nan_debug_info(batch_idx, epoch_idx, ret_dict, recoder)
            torch.cuda.empty_cache()
            
            if nan_count >= 30:
                recoder.print_log("Too many NaN losses, stopping training")
                raise ValueError("Training stopped due to excessive NaN losses")
            continue
        
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer.optimizer)
            scaler.update()
            optimizer.zero_grad()
        loss_item = loss.item()
        
        wandb.log({
            "train/loss": loss_item,
            "train/lr": clr[0],
            "epoch": epoch_idx,
            "batch": batch_idx
        })
        
        loss_value.append(loss.item())
        del ret_dict
        
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f'\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. '
                f'Loss: {loss_item:.8f} LR: {clr[0]:.8f}'
            )
        tqdm_loader.set_postfix({'Loss': loss_item})
        
        if batch_idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
                
    optimizer.scheduler.step()
    mean_loss = np.mean(loss_value) if loss_value else float('inf')
    wandb.log({"train/epoch_mean_loss": mean_loss, "epoch": epoch_idx})
    recoder.print_log(f'\tMean training loss: {mean_loss:.10f}.')
    return mean_loss


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):
    """
    Evaluate model and return metrics dictionary.
    
    Returns:
        dict: Contains 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L' as percentages (0-100 scale)
    """
    model.eval()
    total_sent = []
    total_info = []
    eval_metrics_res = []
    
    for batch_idx, data in enumerate(tqdm(loader, ncols=100)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        gt_sents = [s.split('|')[-1] for s in data[4]]
        
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, gt_sentences=gt_sents)
            
        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
        
        # Compute BLEU and ROUGE scores per batch
        scores = compute_rouge_bleu_batch(
            gt_sents, 
            ret_dict['recognized_sents'], 
            bleu_n=(1, 2, 3, 4)
        )
        eval_metrics_res.extend(scores)
        
        del vid, vid_lgt, label, label_lgt
        
        if batch_idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Write predictions to file
    write2file(
        work_dir + f"output-hypothesis-{mode}.ctm", 
        total_info, 
        total_sent
    )
    
    # Calculate mean scores (already in 0-100 range from compute_rouge_bleu_batch)
    if eval_metrics_res:
        mean_bleu_scores = {
            f'BLEU-{n}': (sum(r[f'BLEU-{n}'] for r in eval_metrics_res) / len(eval_metrics_res)) * 100
            for n in (1, 2, 3, 4)
        }
        mean_rouge = (sum(r['ROUGE-L'] for r in eval_metrics_res) / len(eval_metrics_res)) * 100
        
        metrics = {
            **mean_bleu_scores,
            'ROUGE-L': mean_rouge
        }
    else:
        recoder.print_log(f"Warning: No evaluation results for {mode}")
        metrics = {
            'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0, 'ROUGE-L': 0.0
        }
    
    # Log to wandb
    wandb_log_dict = {f'{mode}/{key}': value for key, value in metrics.items()}
    wandb_log_dict['epoch'] = epoch
    wandb.log(wandb_log_dict)
    
    # Print results (values are already percentages)
    metrics_str = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])
    recoder.print_log(
        f"Epoch {epoch}, {mode} Results -> {metrics_str}", 
        f"{work_dir}/{mode}.txt"
    )
    
    del total_sent, total_info, eval_metrics_res
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return
    
    for batch_idx, data in tqdm(enumerate(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
            
        if not os.path.exists(src_path):
            os.makedirs(src_path)
            
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.savez(filename, save_file)
            start = end
            
        assert start == len(data[2]), "Mismatch in processed samples"
        
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    """Write predictions to CTM format file"""
    with open(path, "w") as f:
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                f.write(
                    f"{info[sample_idx]} 1 {word_idx * 1.0 / 100:.2f} "
                    f"{(word_idx + 1) * 1.0 / 100:.2f} {word[0]}\n"
                )

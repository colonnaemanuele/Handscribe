import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import wandb
import importlib
import inspect
from collections import OrderedDict
from shutil import copytree

from handscribe.modules.sync_batchnorm.batchnorm import convert_model
import handscribe.utils as utils
from handscribe.utils.helper import CheckpointManager, MetricMode
from seq_scripts import seq_train, seq_eval, seq_feature_generation

class ExperimentProcessor:
    def __init__(self, args):
        self.args = args
        self.device = utils.GpuDataParallel()
        self.recoder = None
        self.dataset = {}
        self.data_loader = {}
        self.model = None
        self.optimizer = None
        self.scaler = None
        
    def setup(self):
        """Prepare directories, logging, and W&B"""
        self._prepare_directories()
        self._init_recorder_and_wandb()
        self._set_seed()
        
    def load(self):
        """Load data and model"""
        self.model, self.optimizer, self.scaler = self._load_model()
        self._load_data()
        

    def run(self):
        """Main execution loop based on phase"""
        if self.args.phase == "train":
            self._run_training()
        elif self.args.phase == "eval":
            self._run_evaluation()
        elif self.args.phase == "features":
            self._run_feature_generation()


    def _prepare_directories(self):
        if os.path.exists(self.args.work_dir):
            # Safe clean: only if explicitly asked via some flag logic you prefer
            pass 
        else:
            os.makedirs(self.args.work_dir)
            
        # Copy source code for reproducibility
        shutil.copy2("main.py", self.args.work_dir) # Copia il main
        shutil.copy2("handscribe/configs/baseline/baseline.yaml", self.args.work_dir)
        copytree("handscribe/slowfast_modules", self.args.work_dir + "/slowfast_modules", dirs_exist_ok=True)
        copytree("handscribe/modules", self.args.work_dir + "/modules", dirs_exist_ok=True)

    def _init_recorder_and_wandb(self):
        self.recoder = utils.Recorder(self.args.work_dir, self.args.print_log, self.args.log_interval)
        
        # WandB Setup
        if hasattr(self.args, 'wandb') and isinstance(self.args.wandb, dict):
            w_proj = self.args.wandb.get('wandb_project', 'handscribe')
            w_entity = self.args.wandb.get('wandb_entity', None)
            w_off = self.args.wandb.get('wandb_offline', False)
        else:
            w_proj = getattr(self.args, 'wandb_project', 'handscribe')
            w_entity = getattr(self.args, 'wandb_entity', None)
            w_off = getattr(self.args, 'wandb_offline', False)

        wandb.init(
            project=w_proj,
            entity=w_entity,
            mode='offline' if w_off else 'online',
            config=vars(self.args),
            name=os.path.basename(self.args.work_dir.rstrip('/'))
        )

    def _set_seed(self):
        if self.args.random_fix:
            self.rng = utils.RandomState(seed=self.args.random_seed)
        else:
            self.rng = utils.RandomState(seed=None)

    def _load_data(self):
        print("Loading data...")
        # Gloss dict logic
        if not self.args.gloss_free:
            self.gloss_dict = np.load(self.args.dataset_info["dict_path"], allow_pickle=True).item()
            self.args.model_args["num_classes"] = len(self.gloss_dict) + 1
        else:
            self.gloss_dict = None
            
        feeder_class = self._import_class(self.args.feeder)
        shutil.copy2(inspect.getfile(feeder_class), self.args.work_dir)
        
        # Determine dataset splits based on dataset name
        dataset_list = self._get_dataset_splits()

        for mode, train_flag in dataset_list:
            args = self.args.feeder_args.copy()
            args["prefix"] = self.args.dataset_info["dataset_root"]
            args["mode"] = mode.split("_")[0]
            args["transform_mode"] = train_flag
            
            self.dataset[mode] = feeder_class(
                gloss_free=self.args.gloss_free,
                gloss_dict=self.gloss_dict,
                kernel_size=self.kernel_sizes,
                dataset=self.args.dataset,
                **args
            )
            self.data_loader[mode] = self._build_dataloader(self.dataset[mode], mode, train_flag, feeder_class.collate_fn)
            
    def _get_dataset_splits(self):
        name = self.args.dataset
        if name == "CSL": return zip(["train", "dev"], [True, False])
        if "phoenix" in name or name == "CSL-Daily" or 'lis' in name:
            return zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        return []

    def _build_dataloader(self, dataset, mode, is_train, collate_fn):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size if mode == "train" else self.args.test_batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=self.args.num_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid)
        )

    def _load_model(self):
        print("Loading model...")
        self.device.set_device(self.args.device)
        model_class = self._import_class(self.args.model)
        
        # Fix args structure for model init
        slowfast_args_list = []
        if hasattr(self.args, 'slowfast_args') and self.args.slowfast_args:
            slowfast_dict = vars(self.args.slowfast_args) if hasattr(self.args.slowfast_args, '__dict__') else self.args.slowfast_args
            for k, v in slowfast_dict.items():
                slowfast_args_list.extend([k, v])
            
        model = model_class(
            self.args.model_args,
            loss_weights=self.args.loss_weights,
            load_pkl=not (self.args.load_checkpoints or self.args.load_weights),
            slowfast_config=self.args.slowfast_config,
            slowfast_args=slowfast_args_list,
            language_model=self.args.lang,
            decoder_args=self.args.decoder_args,
        )
        
        shutil.copy2(inspect.getfile(model_class), self.args.work_dir)
        optimizer = utils.Optimizer(model, self.args.optimizer_args)
        scaler = torch.amp.grad_scaler.GradScaler()

        # Load weights/checkpoints
        if self.args.load_checkpoints:
            self._load_checkpoint(model, optimizer, scaler)
        elif self.args.load_weights:
            self._load_weights_only(model, self.args.load_weights)
            
        # Move to GPU
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model = nn.DataParallel(model, device_ids=self.device.gpu_list, output_device=self.device.output_device)
            model = convert_model(model)
        model.cuda()
        
        if hasattr(model, 'conv1d'):
            self.kernel_sizes = model.conv1d.kernel_size 
        return model, optimizer, scaler

    def _run_training(self):
        ckpt_manager = CheckpointManager(self.args.work_dir, keep_best_n=3, recoder=self.recoder)
        start_epoch = self.args.optimizer_args.get("start_epoch", 0)
        
        for epoch in range(start_epoch, self.args.num_epoch):
            # Train
            train_loss = seq_train(self.data_loader["train"], self.model, self.optimizer, 
                                 self.device, epoch, self.recoder, self.scaler)
            
            # Eval
            if epoch % self.args.eval_interval == 0:
                metrics = {'train_loss': train_loss}
                
                # Evaluate on Dev and Test
                for phase in ['dev', 'test']:
                    if phase in self.data_loader:
                        phase_metrics = seq_eval(self.args, self.data_loader[phase], self.model, 
                                               self.device, phase, epoch, self.args.work_dir, 
                                               self.recoder, self.args.evaluate_tool)
                        # Add to main metrics dict with prefix
                        metrics.update({f"{phase}_{k.lower().replace('-','')}" : v for k, v in phase_metrics.items()})
                        
                        # Calculate combined
                        if 'BLEU-4' in phase_metrics and 'ROUGE-L' in phase_metrics:
                            metrics[f'{phase}_combined'] = (phase_metrics['BLEU-4'] + phase_metrics['ROUGE-L']) / 2.0

                self.recoder.print_log(f"Epoch {epoch} Metrics: {metrics}")
                
                # Save Checkpoints
                for metric_name in ['dev_bleu4', 'dev_combined']:
                    if metric_name in metrics:
                        ckpt_manager.update_and_save(epoch, self.model, self.optimizer, 
                                                   self.scaler, self.rng, metrics, metric_name, MetricMode.MAXIMIZE)

    def _run_evaluation(self):
        for mode in ["dev", "test"]:
            if mode in self.data_loader:
                seq_eval(self.args, self.data_loader[mode], self.model, self.device, 
                        mode, 0, self.args.work_dir, self.recoder, self.args.evaluate_tool)

    def _run_feature_generation(self):
        for mode in ["train", "dev", "test"]:
            loader_key = mode + "_eval" if mode == "train" else mode
            if loader_key in self.data_loader:
                seq_feature_generation(self.data_loader[loader_key], self.model, self.device, 
                                     mode, self.args.work_dir, self.recoder)

    def _load_weights_only(self, model, path):
        state_dict = torch.load(path, weights_only=False)
        weights = OrderedDict([(k.replace(".module", ""), v) for k, v in state_dict["model_state_dict"].items()])
        model.load_state_dict(weights, strict=False)
        print("Weights loaded successfully.")

    def _load_checkpoint(self, model, optimizer, scaler):
        # Implementation of full checkpoint loading (state, rng, optimizer, etc.)
        state_dict = torch.load(self.args.load_checkpoints, weights_only=False)
        model.load_state_dict(state_dict["model_state_dict"])
        if "optimizer_state_dict" in state_dict: optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if "scaler_state_dict" in state_dict: scaler.load_state_dict(state_dict["scaler_state_dict"])
        self.args.optimizer_args["start_epoch"] = state_dict["epoch"] + 1

    @staticmethod
    def _import_class(name):
        components = name.rsplit(".", 1)
        mod = importlib.import_module(components[0])
        return getattr(mod, components[1])
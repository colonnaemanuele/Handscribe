import pdb
import torch
import torch.optim as optim
from GFSlowFastSign.utils.cosinelr import CosineAnnealingWarm
from muon import Muon

class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'AdamW':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Muon':
            muon_params = {
                'lr': self.optim_dict['base_lr'],
                'momentum': self.optim_dict.get('momentum', 0.95),
                'nesterov': self.optim_dict.get('nesterov', True),
                'backend': self.optim_dict.get('backend', 'newtonschulz5'),
                'rank': self.optim_dict.get('rank', 1),
            }
            
            if 'weight_decay' in self.optim_dict and self.optim_dict['weight_decay'] > 0:
                try:
                    test_params = list(model.parameters())[:1]
                    test_optimizer = Muon(test_params, weight_decay=0.01, **muon_params)
                    muon_params['weight_decay'] = self.optim_dict['weight_decay']
                except TypeError:
                    print("Warning: This version of Muon doesn't support weight_decay parameter")
            
            self.optimizer = Muon(model.parameters(), **muon_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim_dict['optimizer']}")
        
        self.scheduler = self.define_lr_scheduler(self.optimizer)

    def define_lr_scheduler(self, optimizer):
        print(f"Using {self.optim_dict['scheduler']} scheduler")
        if self.optim_dict["optimizer"] in ['SGD', 'Adam', 'AdamW', 'Muon']:
            if self.optim_dict["scheduler"] == 'cosine':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    self.optim_dict['num_epoch'],
                    self.optim_dict['base_lr'] * 0.025
                )
            elif self.optim_dict["scheduler"] == 'multistep':
                print(f'Using MultiStepLR with milestones: {self.optim_dict["step"]}')
                steps = self.optim_dict["step"]
                lr_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=steps,
                    gamma=self.optim_dict['gamma']
                )
            elif self.optim_dict["scheduler"] == 'warmup_cosine':
                lr_scheduler = CosineAnnealingWarm(
                    optimizer,
                    self.optim_dict['num_epoch'],
                    self.optim_dict['base_lr'],
                    warmup_epochs=self.optim_dict.get('warmup_epochs', 5),
                    warmup_start_lr=self.optim_dict.get('warmup_start_lr', self.optim_dict['base_lr'] * 0.1)
                )
            else:
                raise ValueError(f"Unsupported scheduler: {self.optim_dict['scheduler']}")
            return lr_scheduler
        else:
            raise ValueError(f"Unsupported optimizer for scheduler: {self.optim_dict['optimizer']}")

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def get_lr(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0] if self.scheduler else self.optim_dict['base_lr']

    def step_scheduler(self):
        """Step the learning rate scheduler"""
        if self.scheduler:
            self.scheduler.step()
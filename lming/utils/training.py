import torch
from typing import Dict, List, Tuple
from torch.optim import Adam
import madgrad
from einops import rearrange
import numpy as np
from apex.optimizers import FusedAdam


class CosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, peak_value, final_value):
        self.is_warmup = True
        self.warmup_steps = warmup_steps
        self.peak_value = peak_value
        self.final_value = final_value
        super().__init__(optimizer)
        
    def is_warming_up(self):
        if self.is_warmup:
            return self.last_epoch < self.warmup_steps
        else:
            return False

    def set_cosine_schedule(self, remaining_steps):
        # reset the step to 0
        self.last_epoch = 0
        self.is_warmup = False
        self.steps = remaining_steps

    def get_lr(self):
        if self.is_warmup:
            return [self.peak_value * min(1.0, self.last_epoch / self.warmup_steps) for _ in self.base_lrs]
        else:
            return [self.final_value + 0.5 * (self.peak_value - self.final_value) * (1 + np.cos((self.last_epoch) / (self.steps) * np.pi)) for _ in self.base_lrs]


def load_optimizer(config:Dict, model:torch.nn.Module):
    # check device
    model_device = next(model.parameters()).device.type

    optim_type = config['optimizer']['name']
    allowed_types = ['adam', 'madgrad']
    
    assert optim_type in allowed_types, f'Unknown optimizer {optim_type}, must be one of {allowed_types}'
    assert model_device in ['cpu', 'cuda'], f'Unknown device {model_device}, must be one of [cpu, cuda]'

    optim_args = config['optimizer']['args']

    if optim_type == 'adam':
        optimizer = Adam(model.parameters(), **optim_args) if model_device == 'cpu' else FusedAdam(model.parameters(), **optim_args)
    elif optim_type == 'madgrad':
        optimizer = madgrad.MADGRAD(model.parameters(), **optim_args)

    sheduler = CosineLRScheduler(
        optimizer = optimizer,
        warmup_steps = config['scheduler']['warmup_steps'],
        peak_value = config['optimizer']['args']['lr'],
        final_value = 0, # decay to 0
    )

    return optimizer, sheduler

def token_lens_to_mask(token_lens, max_len=None):
    max_len = token_lens.max() if max_len is None else max_len
    mask = torch.arange(max_len, device=token_lens.device)[None, :] < token_lens[:, None]
    return mask

def mark_padding(targets, mask, pad_id):
    targets[~mask] = pad_id
    return targets

def add_eos(tokens, eos_id, token_lens):
    tokens[torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.long), (token_lens - 1).to(torch.long)] = eos_id 
    return tokens

def loss_ce(logits, labels, ignore_index=-100, label_smoothing=0.0, reduction='sum'):
    return torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'), 
            labels, 
            ignore_index = ignore_index,
            label_smoothing = label_smoothing,
            reduction = reduction
        )


def get_dtype(args):
    if args.dtype == 'fp16':
        dtype = torch.half
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return dtype

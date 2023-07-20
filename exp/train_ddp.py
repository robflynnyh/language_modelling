
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lming.models.transformer import transformer_lm
from omegaconf.omegaconf import OmegaConf

from lming.loading.datasets.MultiDataset import SimpleDistributedDataloader #####!
import traceback
from lming.loading.tokenizer import load_tokenizer

from lming.utils.general import load_model, save_model, load_checkpoint

from einops import rearrange
import numpy as np
import os



import madgrad
import wandb
wandb.login()

from torch.cuda.amp import GradScaler
from torch import autocast

from apex.optimizers import FusedAdam
from torch.optim import Adam

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

def token_lens_to_mask(token_lens, max_len=None):
    max_len = token_lens.max() if max_len is None else max_len
    mask = torch.arange(max_len, device=token_lens.device)[None, :] < token_lens[:, None]
    return mask

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


def setup(rank, worldsize):
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

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
        final_value = 1e-7, # decay to 0
    )

    return optimizer, sheduler

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

def train(
        args:argparse.Namespace,
        model:transformer_lm, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler, 
        device:torch.device,
        skip_to:int = 0,
    ):
    scaler = GradScaler()

    wandb_config = args.config['wandb']

    model.train()
    model_dtype = next(model.parameters()).dtype
    loss_fn = lambda logits, targets: loss_ce(
        logits=logits, 
        labels=targets, 
        ignore_index=-100,
    )

    backprop_every = args.config['training']['backprop_every']

    max_cache_length = args.config['training']['max_seq_len']


    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        # save every 100 steps
        if i % args.config['checkpointing']['save_every_n_steps'] == 0 and i != 0 and args.gpu == 0:
            save_model(model, optimizer, scheduler, i*args.config['training']['batch_size'] + skip_to, args.config)

        chunks = batch

        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                current_recording = i * args.config['training']['batch_size'] 
                total_recordings = len(dataloader) * args.config['training']['batch_size'] 
                remaining_recordings = total_recordings - current_recording
                remaining_steps = remaining_recordings // args.config['training']['batch_size']
                scheduler.set_cosine_schedule(remaining_steps)

        prev_selection_mask = None # selection mask from previous chunk
        last_kv_set = None
 
        for ix, chunk_json in enumerate(chunks):
            print(f'chunk {ix}/{len(chunks)}')
            
            tokens, lengths, selection_mask = chunk_json['tokens'], chunk_json['lengths'], chunk_json['selection_idx']

            cur_selection_mask = None
            if prev_selection_mask != None and not torch.allclose(selection_mask, prev_selection_mask):
                cur_selection_mask = selection_mask[prev_selection_mask]
                
            cur_tokens_in_loss = 0
            cur_loss = 0
            
            tokens, lengths = tokens.to(device, dtype=torch.long), lengths.to(device, dtype=torch.long)
       
            targets = tokens.clone()
            targets[:, :-1] = tokens[:, 1:] # shift right
            if lengths.max() == lengths.min():
                targets[:, -1] = -100 # mask last token
            else:
                targets = add_eos(targets, -100, lengths)
            mask = token_lens_to_mask(lengths) 
            targets = mark_padding(targets, mask, -100)


            with autocast(device.type, dtype=torch.bfloat16):
                cached_kvs = last_kv_set.clone() if last_kv_set != None else None
                cached_kv_lengths = torch.LongTensor([cached_kvs.shape[-2]] * cached_kvs.shape[2]).to(device) if cached_kvs != None else None

                if cur_selection_mask != None and cached_kvs != None:
                    cached_kvs = cached_kvs[:,:,cur_selection_mask]
                    cached_kv_lengths = cached_kv_lengths[cur_selection_mask]
                    
                pred, _, cached_kvs = model(
                    x = tokens, 
                    length = lengths, 
                    cache = None if cached_kvs == None else {
                        'cache': cached_kvs,
                        'cache_lengths': cached_kv_lengths
                    }
                )
                out_kvs = cached_kvs['cache']
                if max_cache_length != 0:
                    last_kv_set = out_kvs[:, :, :, :, -max_cache_length:].clone()
                
                
                B,N,C = pred.shape 
                
                loss = loss_fn(logits=pred, targets=targets) 
            

            cur_loss += loss

            # cur_tokens_in_loss += B * N
            cur_tokens_in_loss += sum(lengths) # total number of acoustic frames in batch

            scaler.scale(loss / (B*args.config['text_chunking']['size'])).backward() # fix by max sequence length so shorter sequences don't get more weight
            last_kv_set.detach_() if last_kv_set != None else None

            if (ix+1) % backprop_every == 0 or ix == len(chunks) - 1:
                full_loss = cur_loss 
                full_loss /= cur_tokens_in_loss
            
                loss_to_log = full_loss.item()
                print(f'loss: {full_loss}')
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                if scheduler.is_warmup:
                    scheduler.step()

                learning_rate = scheduler.get_last_lr()[0]

                if wandb_config['use'] and args.gpu == 0:
                    wandb.log({
                        'loss': loss_to_log,
                        'learning_rate': learning_rate,
                    })

                
                cur_tokens_in_loss = 0
                cur_loss = 0

            prev_selection_mask = selection_mask.clone()

        if not scheduler.is_warmup: # step every batch
            scheduler.step()
        del full_loss, loss_to_log, cur_loss, cur_tokens_in_loss, loss

  

    # save final model
    save_model(model, optimizer, None, i*args.config['training']['batch_size'] + skip_to, args.config)
    return model




def main(gpu, args):
    assert args.world_size > 1, 'must use distributed training for this script, use train.py for single gpu training'
    setup(gpu, args.world_size)

    args.config = OmegaConf.load(args.config)

    checkpoint_dir = args.config['checkpointing']['dir']
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir); print(f'created checkpoint dir: {checkpoint_dir}')

    tokenizer = load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()


    #print(args.config)
    wandb_config = args.config['wandb']
    print(f'running on gpu: {gpu}')
    if gpu == 0: 
        if wandb_config['use']:
            project_name, w_id = wandb_config['project_name'], wandb_config['id']
            wandb.init(project=project_name, config={**args.config}) if w_id == '' else wandb.init(project=project_name, id=w_id, resume="must", config=args.config, allow_val_change=True)
            #wandb.watch(model, log="all") # sometimes this causes a crash ):
            wandb.config.update({'total_params': tparams}, allow_val_change=True)
            print(f'\nLoggging with Wandb id: {wandb.run.id}\n')


    optimizer, scheduler = load_optimizer(args.config, model)
    step = load_checkpoint(args, model, optimizer, scheduler, args.config['checkpointing']['dir'], location='cpu')
    if args.reset_step:
        step = 0

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    model = DDP(model.to(device), device_ids=[gpu], output_device=gpu, gradient_as_bucket_view=True)

    print(f'Starting from podcast: {step}')
    # skip data up to step
    dataloader = SimpleDistributedDataloader(
        tokenizer = tokenizer,
        max_seq_len = args.config.text_chunking['size'],
        batch_size = args.config['training']['batch_size'],
        skip_to = step,
        world_size = args.world_size,
        rank = gpu,
    )
    
    args.gpu = gpu
    final_model = train(args, model, dataloader, optimizer, scheduler, device, skip_to = step)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-rm_sched', '--remove_scheduler', action='store_true', help='remove scheduler from checkpoint')
    parser.add_argument('-reset_step', '--reset_step', action='store_true', help='reset step to 0')
    parser.add_argument('-anomaly', '--anomaly', action='store_true', help='turn on anomaly detection')
    parser.add_argument('-ws', '--world_size', type=int, default=2, help='number of gpus')

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    local_rank = int(os.environ['LOCAL_RANK'])

    main(gpu=local_rank, args=args)
    
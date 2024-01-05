
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lming.models.mamba import MambaLMHeadModel
from omegaconf.omegaconf import OmegaConf

from lming.loading.datasets.PG19Dataset import SimpleDistributedDataloader #####!
import traceback
from lming.loading.tokenizer import load_tokenizer
import datetime
from lming.utils.general import load_model, save_model, load_checkpoint, optimizer_to

from lming.utils.training import (
    load_optimizer, 
    token_lens_to_mask, 
    mark_padding,
    add_eos,
    loss_ce,
    get_dtype
)

from einops import rearrange
import numpy as np
import os

import wandb
wandb.login()

from torch.cuda.amp import GradScaler
from torch import autocast

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

from mamba_ssm.utils.generation import InferenceParams


def setup(rank, worldsize):
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

def cleanup():
    dist.destroy_process_group()

def drop_inference_cache(cache, selection_mask, B):
    if selection_mask is not None:
        for key in cache.key_value_memory_dict.keys():
            cache.key_value_memory_dict[key] = (cache.key_value_memory_dict[key][0].detach().clone()[selection_mask], cache.key_value_memory_dict[key][1].detach().clone()[selection_mask])
         
    return cache


def train(
        args:argparse.Namespace,
        model:MambaLMHeadModel, 
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

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        # save every 100 steps
        if i % args.config['checkpointing']['save_every_n_steps'] == 0 and i != 0:
            save_model(model, optimizer, scheduler, i*args.config['training']['batch_size'] + skip_to, args.config)

        chunks = batch
        # shuffle chunks
        #np.random.shuffle(chunks)

        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                current_recording = i * args.config['training']['batch_size'] 
                total_recordings = len(dataloader) * args.config['training']['batch_size'] 
                remaining_recordings = total_recordings - current_recording
                remaining_steps = remaining_recordings // args.config['training']['batch_size']
                scheduler.set_cosine_schedule(remaining_steps)

        cur_tokens_in_loss, cur_loss = 0, 0
        cache = InferenceParams(max_seqlen=np.inf, max_batch_size=np.inf)
        prev_selection_mask = None # selection mask from previous chunk


        for ix, chunk_json in enumerate(chunks):
            print(f'chunk {ix}/{len(chunks)}')
            
            tokens, lengths, selection_mask = chunk_json['tokens'], chunk_json['lengths'], chunk_json['selection_idx']

            cur_selection_mask = None
            if prev_selection_mask != None and not torch.allclose(selection_mask, prev_selection_mask):
                cur_selection_mask = selection_mask[prev_selection_mask]

            # drop tokens with length smaller than 12
            #tokens, cur_selection_mask, lengths  = tokens[lengths >= 8], cur_selection_mask[lengths >= 8], lengths[lengths >= 8]# some bug in s6 code
            # UPDATE MAMBA CODE AND REMOVE ABOVE LINE OR, ALSO DROP THE SELECTION MASK WHERE LENGTHS < 12
            if lengths.numel() == 0:
                continue # skip empty batch
            
            tokens, lengths = tokens.to(device, dtype=torch.long), lengths.to(device, dtype=torch.long)

            cache = drop_inference_cache(cache, cur_selection_mask, tokens.shape[0]) 

            targets = tokens.clone()
            targets[:, :-1] = tokens[:, 1:] # shift right
            if lengths.max() == lengths.min():
                targets[:, -1] = -100 # mask last token
            else:
                targets = add_eos(targets, -100, lengths)

            mask = token_lens_to_mask(lengths) 
            targets = mark_padding(targets, mask, -100)

            with autocast(device.type, dtype=get_dtype(args)):
                pred = model(input_ids = tokens.contiguous(), inference_params=cache).logits
                B,N,C = pred.shape 
                loss = loss_fn(logits=pred, targets=targets) 

            cur_loss += loss
            cur_tokens_in_loss += sum(lengths) # total number of acoustic frames in batch
            scaler.scale(loss / (B*args.config['text_chunking']['size'])).backward() # fix by max sequence length so shorter sequences don't get more weight

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
                cur_tokens_in_loss, cur_loss = 0, 0
            prev_selection_mask = selection_mask.clone()

        if not scheduler.is_warmup: # step every batch
            scheduler.step()

        del full_loss, loss_to_log, cur_loss, cur_tokens_in_loss, loss

    # save final model
    if args.gpu == 0:
        save_model(model, optimizer, None, i*args.config['training']['batch_size'] + skip_to, args.config)
        print(f'saved final model and finished training')

    return model




def main(gpu, args):
    #assert args.world_size > 1, 'must use distributed training for this script, use train.py for single gpu training'
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
            run_name = None if 'name' not in wandb_config else wandb_config['name']
            wandb.init(project=project_name, config={**args.config}, name=run_name) if w_id == '' else wandb.init(project=project_name, id=w_id, resume="must", config=args.config, allow_val_change=True)
            #wandb.watch(model, log="all") # sometimes this causes a crash ):
            wandb.config.update({'total_params': tparams}, allow_val_change=True)
            print(f'\nLoggging with Wandb id: {wandb.run.id}\n')

        
    optimizer, scheduler = load_optimizer(args.config, model)

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    
    model = DDP(model.to(device), device_ids=[gpu], output_device=gpu, gradient_as_bucket_view=True)

    step = load_checkpoint(args, model, optimizer, scheduler, args.config['checkpointing']['dir'], location=lambda storage, loc: storage)
    torch.cuda.empty_cache()
    if args.reset_step:
        step = 0
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
    print(f'finished training on gpu: {gpu}')

    cleanup()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-rm_sched', '--remove_scheduler', action='store_true', help='remove scheduler from checkpoint')
    parser.add_argument('-reset_step', '--reset_step', action='store_true', help='reset step to 0')
    parser.add_argument('-anomaly', '--anomaly', action='store_true', help='turn on anomaly detection')
    parser.add_argument('-ws', '--world_size', type=int, default=2, help='number of gpus')
    parser.add_argument('-dtype', '--dtype', type=str, default='bf16')

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    local_rank = int(os.environ['LOCAL_RANK'])

    main(gpu=local_rank, args=args)
    
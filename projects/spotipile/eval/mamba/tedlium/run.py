import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from omegaconf.omegaconf import OmegaConf
import sentencepiece as spm
from lming.models.transformer import transformer_lm
from lming.utils.general import load_model, save_model, load_checkpoint


from lming.loading.tokenizer import load_tokenizer
import pandas as pd
from einops import rearrange
import os
import wandb
from collections import namedtuple
import re
from tqdm import tqdm

DEV_PATH = '/store/store4/data/TEDLIUM_release1/legacy/dev/'
TEST_PATH = '/store/store4/data/TEDLIUM_release1/legacy/test/'
TRAIN_EXAMPLE = '/store/store4/data/TEDLIUM_release1/legacy/train/stm/EllenGustafson_2010X.stm'


def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def proc_stm_and_timings(stm_path:str):
    stm = open_stm(stm_path)
    all_text = ""
    timings = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            continue
        all_text += text + ' '
        timings.append({'start': float(start), 'end': float(end)})
    all_text = all_text.strip()
    # regex to do all of the above
    # i.e replace space followed by a apostrophe followed by a letter with just the apostrophe and letter
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    # remomve anything inside of any brackets
    all_text = re.sub(r"\(.*?\)", "", all_text)
    all_text = re.sub(r"\[.*?\]", "", all_text)
    all_text = re.sub(r"\{.*?\}", "", all_text)
    all_text = re.sub(r"\<.*?\>", "", all_text)
    # remove multiple spaces
    all_text = re.sub(r" +", r" ", all_text)
    return all_text, timings

def fetch_data(path:str = TEST_PATH):
    audio_path = os.path.join(path, 'sph')
    audio_files = [os.path.join(audio_path, el) for el in os.listdir(audio_path) if el.endswith('.sph')]
    audio_files.sort()
    text_path = os.path.join(path, 'stm')
    text_files = [os.path.join(text_path, el) for el in os.listdir(text_path) if el.endswith('.stm')]
    text_files.sort()
    assert len(audio_files) == len(text_files), 'Number of audio files and text files must match'
    return audio_files, text_files

def loss_ce(logits, labels, ignore_index=-100, label_smoothing=0.0, reduction='sum'):
    return torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'), 
            labels, 
            ignore_index = ignore_index,
            label_smoothing = label_smoothing,
            reduction = reduction
        )

@torch.no_grad()
def get_perplexity(args:argparse.Namespace, model:transformer_lm, text:str, tokenizer:spm.SentencePieceProcessor, prev_cache:Dict[str, torch.Tensor] = None):
    tokenized_text = tokenizer.encode(text)
    bos = tokenizer.bos_id()
    tokenized_text = [bos] + tokenized_text
    seq_len = args.seq_len //2 if args.seq_len != -1 else args.config['text_chunking']['size'] // 2
    cache_len = seq_len

    print(f'Processing with seq_len: {seq_len} and cache_len: {cache_len}')

    loss_fn = lambda logits, targets: loss_ce(
        logits=logits, 
        labels=targets, 
        ignore_index=-100,
    )

    all_logits = []
    # process text in chunks of seq_len
    inference_params = None # look at:https://github.com/state-spaces/mamba/blob/bae8d1a42fec58f4cdd300bf3b987d05eab22ed0/mamba_ssm/utils/generation.py#L18

    pbar = tqdm(range(0, len(tokenized_text), seq_len), total=len(tokenized_text)//seq_len)
    for i in pbar:
        cur_chunk = tokenized_text[i:i+seq_len]
        cur_chunk = torch.LongTensor(cur_chunk).unsqueeze(0).to(model.device)
   
        logits = model(
            input_ids = cur_chunk, 
        ).logits

        all_logits.append(logits)
    
    pbar.close()
    all_logits = torch.cat(all_logits, dim=1)
    target = torch.LongTensor(tokenized_text)[None, 1:].to(model.device)    
    all_logits = all_logits[:, :-1]
    
    loss = loss_fn(all_logits, target) # reduyction is sum
    total_words = len(text.split(' ')) - 1 # -1 bcos last word is not predicted
    perplexity = torch.exp(loss / total_words)
    print(f'Perplexity: {perplexity.item()}, Total words: {total_words}, Total Tokens: {len(tokenized_text)}')
    return loss, total_words, len(tokenized_text)

@torch.no_grad()
def get_init_seq(args:argparse.Namespace, model:transformer_lm, text:str, tokenizer:spm.SentencePieceProcessor):
    text = text + '. ' if not text.endswith('.') else text + ' '
    tokenized_text = tokenizer.encode(text + '. ') # add a period to the end
    bos = tokenizer.bos_id()
    tokenized_text = [bos] + tokenized_text
    seq_len = args.seq_len if args.seq_len != -1 else args.config['text_chunking']['size']
    cache_len = args.cache_len if args.cache_len != -1 else args.config['training']['max_seq_len']

    print(f'Processing with seq_len: {seq_len} and cache_len: {cache_len}')

    all_logits = []
    # process text in chunks of seq_len
    prev_cache = None
    pbar = tqdm(range(0, len(tokenized_text), seq_len), total=len(tokenized_text)//seq_len)
    for i in pbar:
        cur_chunk = tokenized_text[i:i+seq_len]
        cur_chunk = torch.LongTensor(cur_chunk).unsqueeze(0).to(model.device)
   
        logits, _, cached_kvs = model(x = cur_chunk, cache = prev_cache)
        all_logits.append(logits)
        if cache_len != 0:
            prev_cache = cached_kvs
            prev_cache['cache'] = prev_cache['cache'][:, :, :, :, -cache_len:]
            prev_cache['cache_lengths'] = prev_cache['cache_lengths'] * 0 + prev_cache['cache'].shape[-2]

    pbar.close()
    return prev_cache

def convert_from_ddp(model_state_dict):
    '''
    Convert model state dict from DDP to single GPU.
    '''
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict

def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if args.from_ddp:
        checkpoint['model'] = convert_from_ddp(checkpoint['model'])
        #torch.save(checkpoint, args.checkpoint)

    model_config = checkpoint['config']
    print(model_config)
    args.config = model_config

    tokenizer = load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    model = model.to(device)
    model.device = device
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded model from {args.checkpoint}')
    model.print_total_params()
    model.eval()

    assert args.split in ['dev', 'test'], 'Split must be dev or test'
    data_path = TEST_PATH if args.split == 'test' else DEV_PATH
    print(f'Evaluating {args.split} data')
    _, text_files = fetch_data(path=data_path)

    init_text, _ = proc_stm_and_timings(TRAIN_EXAMPLE)
    prev_cache = get_init_seq(args = args, model = model, text = init_text, tokenizer = tokenizer) if args.buffer else None

    loss_sum, total_words_sum, tokens_sum = 0, 0, 0
    for text_file in text_files:
        print(f'Processing {text_file}')
        text, _ = proc_stm_and_timings(text_file)

        loss, total_words, total_tokens = get_perplexity(
            args = args, 
            model = model, 
            text = text, 
            tokenizer = tokenizer, 
            prev_cache = {
                'cache': prev_cache['cache'].clone(),
                'cache_lengths': prev_cache['cache_lengths'].clone(),
            } if prev_cache is not None else None
        )
        tokens_sum += total_tokens
        loss_sum += loss
        total_words_sum += total_words

    perplexity = torch.exp(loss_sum / tokens_sum)
    print(f'Total Words: {total_words_sum}, Total Tokens: {tokens_sum}')
    print(f'\n\n -----------------\nOverall Perplexity (TOKEN LEVEL): {perplexity.item()}')
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', '--split', type=str, default='test', help='dev or test')
    parser.add_argument('-c', '--checkpoint', type=str, default='/exp/exp4/acp21rjf/checkpoints/language_modelling_spotipile/6e4_ddp/step_684000.pt', help='path to checkpoint')
    parser.add_argument('-fddp', '--from_ddp', action='store_true', help='convert model from DDP to single GPU')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-cache', '--cache_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-buffer','--buffer', action='store_true', help='use buffer') 
    args = parser.parse_args()
    main(args)
    

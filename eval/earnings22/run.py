import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from omegaconf.omegaconf import OmegaConf
import sentencepiece as spm
from lming.models.transformer import transformer_lm
from lming.utils.general import load_model, save_model, load_checkpoint

from lming.loading.tokenizer import load_tokenizer
import json

from einops import rearrange
import os
import wandb
import re
from tqdm import tqdm

TEST_PATH = '/store/store4/data/earnings22/full_transcripts.json'


def parse(txts:List[str]):
    to_remove = ['<inaudible>', '<laugh>'
                    '<noise>', '<silence>', '<unk>', '<vocalized-noise>']
    for r in to_remove:
        txts = [txt.replace(r, '') for txt in txts]
    return txts

def fetch_test_data(path:str = TEST_PATH, return_first_n:int = 10) -> List[str]:
    # # load json file:
    with open(path, 'r') as f:
        data = json.load(f)
    return list(data.values())[:return_first_n]


def loss_ce(logits, labels, ignore_index=-100, label_smoothing=0.0, reduction='sum'):
    return torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'), 
            labels, 
            ignore_index = ignore_index,
            label_smoothing = label_smoothing,
            reduction = reduction
        )

@torch.no_grad()
def get_perplexity(args:argparse.Namespace, model:transformer_lm, text:str, tokenizer:spm.SentencePieceProcessor):
    tokenized_text = tokenizer.encode(text)
    bos = tokenizer.bos_id()
    tokenized_text = [bos] + tokenized_text
    # remove any unk tokens
    tokenized_text = [t for t in tokenized_text if t != tokenizer.unk_id()]
    seq_len = args.seq_len if args.seq_len != -1 else args.config['text_chunking']['size']
    cache_len = args.cache_len if args.cache_len != -1 else args.config['training']['max_seq_len']

    loss_fn = lambda logits, targets: loss_ce(
        logits=logits, 
        labels=targets, 
        ignore_index=-100,
    )

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
    all_logits = torch.cat(all_logits, dim=1)
    target = torch.LongTensor(tokenized_text)[None, 1:].to(model.device)    
    all_logits = all_logits[:, :-1]
    
    loss = loss_fn(all_logits, target) # reduyction is sum
   
    total_words = len(tokenized_text) - 1
    perplexity = torch.exp(loss / total_words)
    print(f'Perplexity: {perplexity.item()}')
    return loss, total_words




def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
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

    text_files = parse(fetch_test_data())



    loss_sum, total_words_sum = 0, 0
    for i, text in enumerate(text_files):
        print(f'Processing {i+1}/{len(text_files)}')
        
        loss, total_words = get_perplexity(args = args, model = model, text = text, tokenizer = tokenizer)
        loss_sum += loss
        total_words_sum += total_words

    perplexity = torch.exp(loss_sum / total_words_sum)
    print(f'\n\n -----------------\nOverall Perplexity: {perplexity.item()}')
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='/exp/exp4/acp21rjf/checkpoints/language_modelling_spotify/1e4/step_105360.pt', help='path to checkpoint')

    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-cache', '--cache_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')

    args = parser.parse_args()
    main(args)
    

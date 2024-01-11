from lming.loading.datasets.PG19Dataset import PG19TestDataset
from lming.loading.tokenizer import load_tokenizer
from lming.utils.general import load_model, get_project_abs_path, convert_from_ddp
import os
import torch
import argparse
from lming.utils.training import loss_ce
from mamba_ssm.utils.generation import InferenceParams
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def calc_perplexity(model, tokenized_file, seq_len, device='cuda'):
    '''
    TODO: currently we move stuff to cpu because of the extra memory allocation created when concatenating the logits
    however if we pre-allocate a tensor for the logits and that just add logits to it at the relevant positions, we can avoid 2 copies of the same tokens
    alternatively could calculate loss in chunks but will have to wait to the next chunk to get the final token (this might be better)
    '''
    loss_fn = lambda logits, targets: loss_ce(
        logits=logits, 
        labels=targets, 
        ignore_index=-100,
    )
    pbar = tqdm(tokenized_file, total=len(tokenized_file))
    all_logits = []
    cache = InferenceParams(max_seqlen=np.inf, max_batch_size=np.inf,seqlen_offset=1)
    for i, chunk in enumerate(pbar):
        chunk = chunk.to(device)
        logits = model(input_ids = chunk[None], inference_params=cache).logits
        all_logits.append(logits.to('cpu'))
        # for k in cache.key_value_memory_dict.keys():
        #     if k == 0:
        #         print(cache.key_value_memory_dict[k][0][0][0])
            #cache.key_value_memory_dict[k] = (cache.key_value_memory_dict[k][0] + 100000, cache.key_value_memory_dict[k][1] - 100)
            #print(cache.key_value_memory_dict[k][0].shape, cache.key_value_memory_dict[k][1].shape)
        # print(cache)
    pbar.close()
    
    all_logits = torch.cat(all_logits, dim=1)
    targets = torch.cat(tokenized_file, dim=0)[None, 1:].to('cpu')
    all_logits = all_logits[:, :-1]

    loss = 0
    for i in range(0, all_logits.shape[1], seq_len): # compute in chunks of seq_len to avoid OOM on long documents
        loss += loss_fn(all_logits[:, i:i+seq_len].to(device), targets[:, i:i+seq_len].to(device))
    
    total_tokens = targets.numel()
    perplexity = torch.exp(loss / total_tokens)
    print(f'Perplexity: {perplexity.item()}, Total Tokens: {total_tokens}')
    return loss, total_tokens

def main(args):
    tokenizer = load_tokenizer(model_path = os.path.join(get_project_abs_path(), './artifacts/pg19tokenizer/tokenizer.model'))
    test_data = PG19TestDataset(tokenizer=tokenizer, max_seq_len=8192, bos_token_id=0)
    cpt = torch.load(args.checkpoint, map_location='cpu')
    model = load_model(cpt['config'], tokenizer.vocab_size())
    model.load_state_dict(convert_from_ddp(cpt['model']))
    model.eval()
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    loss_sum, tokens_sum = 0, 0
    for i, tokenized_file in enumerate(test_data):
        print(f'Processing {i+1}/{len(test_data)}')
        loss, n_tokens = calc_perplexity(model, tokenized_file, seq_len=args.seq_len, device=device)
        loss_sum += loss
        tokens_sum += n_tokens
    overall_perplexity = torch.exp(loss_sum / tokens_sum)
    print(f'Overall Perplexity: {overall_perplexity.item()} Total Tokens: {tokens_sum}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq_len', type=int, default=8192, help='max sequence length')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
    
    args = parser.parse_args()
    main(args)
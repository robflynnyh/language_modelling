import torch
import pandas as pd
import os
from lming.utils.general import load_txt
from lming.loading.tokenizer import load_tokenizer
import random
import sentencepiece as spm
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data.distributed import DistributedSampler
from lming.utils.general import fetch_paths

paths = fetch_paths()
assert 'datasets' in paths, 'Please add datasets path to fetch_paths()'
default_pg19_df_path = paths.datasets.spotify.df_path
default_pg19_base_dir = paths.datasets.spotify.base_dir


def load_pg19(pg19_paths_csv, pg19_base_dir):
    df = pd.read_csv(pg19_paths_csv)
    df['filename'] = df['filename'].apply(lambda x: os.path.join(pg19_base_dir, x))
    df.rename(columns={'filename':'full_path'}, inplace=True)
    df.rename(columns={'total_characters':'length'}, inplace=True)
    return df


class PG19Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path = default_pg19_df_path,
        base_dir = default_pg19_base_dir,
        tokenizer:spm.SentencePieceProcessor = None,
        max_seq_len:int = 1024,
        batch_size:int = 64,
        subgroup_shuffle_size:int = 25000,
        bos_token_id:int = 0,
        skip_to:int = 0,
        random_seed:int = 1234,
    ):
        self.all_df = load_pg19(df_path, base_dir)
        self.base_dir = base_dir
    
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.batch_size = batch_size
        self.bos_token_id = bos_token_id
        self.random_seed = random_seed

        print(f'Total Characters (B): {self.all_df["length"].sum() / 1e9:.2f}')

        self.create_batches()
        self.items = self.items[skip_to:]

    def create_batches(self):
        np.random.seed(self.random_seed), random.seed(self.random_seed)
        self.items = []
        self.all_df = self.all_df.sort_values(by='length') # sort all_df by length min->max
        indices = np.arange(len(self.all_df))   
        indices = [np.random.permutation(indices[i:i+self.subgroup_shuffle_size]) for i in range(0, len(indices), self.subgroup_shuffle_size)]
        indices = np.concatenate(indices)
        indices = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        np.random.shuffle(indices)
        indices = np.concatenate(indices)
        for i in indices:
            self.items.append(self.all_df.iloc[i])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text = load_txt(item['full_path'])
        
        if self.tokenizer is None:
            return text
        
        tokens = self.tokenizer.encode(text)
        tokens = [self.bos_token_id] + tokens
        tokens = [torch.LongTensor(tokens[i:i+self.max_seq_len]) for i in range(0, len(tokens), self.max_seq_len)] # chunk into max_seq_len
        return tokens

def collate_fn(batch):
    #return batch
    chunks = []
    chunk_lens = [len(x) for x in batch]
    max_chunk_len = max(chunk_lens)
    selection_idx = torch.ones(len(batch), dtype=torch.bool)
    for i in range(max_chunk_len):
        chunk_samples = []
        seq_lens = []
        cur_selection_idx = selection_idx.clone()
        for j in range(len(batch)):
            if i < len(batch[j]):
                chunk_samples.append(batch[j][i])
                seq_lens.append(batch[j][i].shape[0])
            else:
                cur_selection_idx[j] = False
        tokens = torch.stack(chunk_samples) if min(seq_lens) == max(seq_lens) else torch.nn.utils.rnn.pad_sequence(chunk_samples, batch_first=True)
        chunks.append({
            'tokens':tokens,
            'selection_idx':cur_selection_idx[selection_idx],
            'lengths':torch.LongTensor(seq_lens),
        })
    return chunks
 
        

class SimpleDataloader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        tokenizer:spm.SentencePieceProcessor = load_tokenizer(),
        df_path = default_pg19_base_dir,
        base_dir = default_pg19_df_path,
        max_seq_len:int = 1024,
        batch_size:int = 64,
        subgroup_shuffle_size:int = 25000,
        bos_token_id:int = 0,
        skip_to:int = 0,
        num_workers = 0,
        pin_memory = False,
    ):
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = PG19Dataset(
            df_path = df_path,
            base_dir = base_dir,
            tokenizer = tokenizer,
            max_seq_len = max_seq_len,
            batch_size = batch_size,
            subgroup_shuffle_size = subgroup_shuffle_size,
            bos_token_id = bos_token_id,
            skip_to = skip_to,
        )
        super().__init__(
                self.dataset, 
                batch_size = batch_size, 
                shuffle = False, 
                num_workers = num_workers,
                pin_memory = pin_memory, 
                collate_fn = collate_fn,
            )


class SimpleDistributedDataloader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        tokenizer:spm.SentencePieceProcessor = load_tokenizer(),
        df_path = default_pg19_df_path,
        base_dir = default_pg19_base_dir,
        max_seq_len:int = 1024,
        batch_size:int = 64,
        subgroup_shuffle_size:int = 3000,
        bos_token_id:int = 0,
        skip_to:int = 0,
        num_workers = 0,
        pin_memory = False,
        world_size = 1,
        rank = 0,
    ):
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = PG19Dataset(
            df_path  = df_path,
            base_dir = base_dir,
            tokenizer = tokenizer,
            max_seq_len = max_seq_len,
            batch_size = batch_size * world_size,
            subgroup_shuffle_size = subgroup_shuffle_size,
            bos_token_id = bos_token_id,
            skip_to = skip_to,
        )
        super().__init__(
                self.dataset, 
                batch_size = batch_size,
                shuffle = False, 
                num_workers = num_workers,
                pin_memory = pin_memory, 
                collate_fn = collate_fn,
                sampler =  DistributedSampler(
                    self.dataset,
                    num_replicas = world_size,
                    rank = rank,
                    shuffle = False,
                )
            )
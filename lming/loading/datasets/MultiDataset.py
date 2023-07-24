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

def load_spotify(podcast_paths_csv, spotify_base_dir):
    df = pd.read_csv(podcast_paths_csv)
    df['podcast_path'] = df['podcast_path'].apply(lambda x: os.path.join(spotify_base_dir, x))
    df.rename(columns={'podcast_path':'full_path'}, inplace=True)
    return df

def load_pile(pile_df_path, pile_base_dir):
    df = pd.read_csv(pile_df_path, header=None)
    df.columns = ["corpus", "full_path", "length"]
    df['full_path'] = df['full_path'].apply(lambda x: os.path.join(pile_base_dir, x))
    return df

class MultiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        spotify_df_path = '/store/store4/data/spotify_text/spotify_podcast_paths.csv',
        spotify_base_dir = '/store/store4/data/spotify_text/podcast_txt',
        pile_df_path = '/store/store4/data/thepile/parsed_data.csv',
        pile_base_dir = '/store/store4/data/thepile/',
        just_spotify = False,
        spotify_upsample = 2.0,
        tokenizer:spm.SentencePieceProcessor = None,
        max_seq_len:int = 1024,
        batch_size:int = 64,
        subgroup_shuffle_size:int = 25000,
        bos_token_id:int = 0,
        skip_to:int = 0,
    ):
        self.spotify_df = load_spotify(spotify_df_path, spotify_base_dir)
        self.spotify_base_dir = spotify_base_dir
        self.pile_df = load_pile(pile_df_path, pile_base_dir) if not just_spotify else None
        self.pile_base_dir = pile_base_dir if not just_spotify else None

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.batch_size = batch_size
        self.bos_token_id = bos_token_id

        self.spotify_df = self.spotify_df.sample(frac=spotify_upsample, replace=True, random_state=1234).reset_index(drop=True)
        self.all_df = pd.concat([self.spotify_df, self.pile_df], ignore_index=True) if not just_spotify else self.spotify_df

        print(f'Total Words (B): {self.all_df["length"].sum() / 1e9:.2f}')

        self.create_batches()
        self.items = self.items[skip_to:]

    def create_batches(self):
        np.random.seed(1234), random.seed(1234)
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
 
        

    return batch

        # spotify_df_path = '/store/store4/data/spotify_text/spotify_podcast_paths.csv',
        # spotify_base_dir = '/store/store4/data/spotify_text/podcast_txt',
        # tokenizer:spm.SentencePieceProcessor = None,
        # max_seq_len:int = 1024,
        # batch_size:int = 64,
        # subgroup_shuffle_size:int = 3000,
        # bos_token_id:int = 0,
class SimpleDataloader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        tokenizer:spm.SentencePieceProcessor = load_tokenizer(),
        spotify_df_path = '/store/store4/data/spotify_text/spotify_podcast_paths.csv',
        spotify_base_dir = '/store/store4/data/spotify_text/podcast_txt',
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
        self.dataset = MultiDataset(
            spotify_df_path = spotify_df_path,
            spotify_base_dir = spotify_base_dir,
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
        spotify_df_path = '/store/store4/data/spotify_text/spotify_podcast_paths.csv',
        spotify_base_dir = '/store/store4/data/spotify_text/podcast_txt',
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
        self.dataset = MultiDataset(
            spotify_df_path = spotify_df_path,
            spotify_base_dir = spotify_base_dir,
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
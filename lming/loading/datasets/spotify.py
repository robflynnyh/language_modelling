import torch
import pandas as pd
import os
from lming.utils.general import load_txt
import sentencepiece as spm
import numpy as np

def load_spotify(podcast_paths_csv, spotify_base_dir):
    df = pd.read_csv(podcast_paths_csv)
    df['full_path'] = df['podcast_path'].apply(lambda x: os.path.join(spotify_base_dir, x))
    df = df.drop('podcast_path', axis=1) 
    return df

class Multi_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        spotify_df = '/store/store4/data/spotify_text/spotify_podcast_paths.csv',
        spotify_base_dir = '/store/store4/data/spotify_text/podcast_txt',
        tokenizer:spm.SentencePieceProcessor = None,
        max_seq_len:int = 1024,
        batch_size:int = 64,
        subgroup_shuffle_size:int = 3000,
        bos_token_id:int = 0,
    ):
        self.spotify_df = spotify_df
        self.spotify_base_dir = spotify_base_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.batch_size = batch_size
        self.bos_token_id = bos_token_id

        self.all_df = self.spotify_df # can add more datasets here in the future
        self.create_batches()

    def create_batches(self):
        np.random.seed(420), pd.np.random.seed(420)
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
        tokens = [tokens[i:i+self.max_seq_len] for i in range(0, len(tokens), self.max_seq_len)] # chunk into max_seq_len
        return tokens






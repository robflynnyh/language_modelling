# this py assumes we have a text file with each line text from a spotify podcast
# we want to create a dataframe where each row is the number of words in a podcast and then the columns is the text files name

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

BASE_DIR = '/store/store4/data/spotify_text/'
ALL_TEXT = 'all_text.txt'
PODCAST_TXT_PATH = 'podcast_txt'
OUTPATH = 'spotify_podcast_paths.csv'

def save_podcast_txt(podcast_id, text):
    with open(os.path.join(BASE_DIR, PODCAST_TXT_PATH, f'podcast_{podcast_id}.txt'), 'w') as f:
        f.write(text)

if not os.path.exists(os.path.join(BASE_DIR, PODCAST_TXT_PATH)):
    os.mkdir(os.path.join(BASE_DIR, PODCAST_TXT_PATH))

# read in all text
with open(os.path.join(BASE_DIR, ALL_TEXT), 'r') as f:
    all_text = f.read().split('\n')

df = []
for i, text in enumerate(tqdm(all_text, total=len(all_text))):
    save_podcast_txt(i, text)
    df.append({
        'podcast_path': f'podcast_{i}.txt',
        'length': len(text.split(' '))
    })

df = pd.DataFrame(df)
df.to_csv(os.path.join(BASE_DIR, OUTPATH), index=False)




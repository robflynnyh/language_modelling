import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lming.models.transformer import transformer_lm
from omegaconf.omegaconf import OmegaConf

from lcasr.loading.datasets.MultiDataset import SimpleDataloader #####!
import traceback

from lming.utils.general import load_model, save_model, load_checkpoint

from einops import rearrange
import numpy as np
import os

import madgrad
import wandb

from torch.cuda.amp import GradScaler
from torch import autocast

from apex.optimizers import FusedAdam
from torch.optim import Adam

import warnings
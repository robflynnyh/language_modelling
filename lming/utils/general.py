import torch, os
from typing import Dict, List, Tuple
from lming.models.transformer import transformer_lm
from lming.models.hyena import HyenaLM
from lming.models.lru import LruLM
from lming.models.mamba import MambaLMHeadModel
from lming.models.meta_transformer import MetaTransformer
from omegaconf import OmegaConf
import sentencepiece as spm

def load_txt(path:str) -> str:
    with open(path, 'r') as f:
        text = f.read() 
    return text
    
def load_model(config:Dict, vocab_size):
    architecture = config.get('architecture', 'transformer')
    if architecture == 'transformer':
        model = transformer_lm(**config.model, vocab_size=vocab_size)
    elif architecture == 'hyena':
        model = HyenaLM(**config.model, vocab_size=vocab_size)
    elif architecture == 'lru':
        model = LruLM(**config.model, vocab_size=vocab_size)
    elif architecture == 'mamba':
        model = MambaLMHeadModel(**config.model, vocab_size=vocab_size)
    elif architecture == 'meta_transformer':
        model = MetaTransformer(**config.model, vocab_size=vocab_size)
    else:
        raise NotImplementedError(f'architecture {architecture} not implemented :(')
    return model

def save_model(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler,
        podcast_step:int,
        config:Dict,
    ):
    save_path = os.path.join(config['checkpointing']['dir'], f'step_{podcast_step}.pt')
    save_dict = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict() if scheduler is not None else None,
        'podcast_step':podcast_step,
        'config':config,
    }
    torch.save(save_dict, save_path)

def find_latest_checkpoint(path:str = './checkpoints'):
    checkpoints = [el for el in os.listdir(path) if el.endswith('.pt')]
    if len(checkpoints) == 0:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return checkpoints[-1]

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

def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def fetch_paths():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    rel_path = '../../paths.yaml'
    path = os.path.join(cur_path, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Could not find paths file at {path} please create one based on paths_template.yaml')
    paths = OmegaConf.load(path)
    return paths
    


def load_checkpoint(args, model, optimizer=None, scheduler=None, path='./checkpoints', location='cpu'):
    latest_checkpoint = find_latest_checkpoint(path) if not path.endswith('.pt') else path
    if latest_checkpoint is None:
        return 0
    path = os.path.join(path, latest_checkpoint)
    checkpoint = torch.load(path)
    #checkpoint['model'] = convert_from_ddp(checkpoint['model'])
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        print('loading model with strict=False')
        model.load_state_dict(checkpoint['model'], strict=False, map_location=location)
        print('SETTING OPTIMIZER TO NONE DUE TO NON-STRICT LOAD'),
        optimizer = None
    print(f'loaded model from {path}')
    if optimizer != None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler != None and 'scheduler' in checkpoint and checkpoint['scheduler'] != None:
        scheduler.load_state_dict(checkpoint['scheduler'])
  
    step = checkpoint['podcast_step']
    return step

def train_tokenizer(
        raw_txt:str = '/mnt/parscratch/users/acp21rjf/spotify/all_text.txt',
        save_path:str = '/mnt/parscratch/users/acp21rjf/spotify/',
        vocab_size:int = 4095,
        normalization_rule_name:str = 'nmt_nfkc_cf',
    ):
    spm.SentencePieceTrainer.train(
        input=raw_txt,
        model_prefix='tokenizer',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        max_sentence_length=1000000, #
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=-1,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        normalization_rule_name=normalization_rule_name
    )
    os.system(f'mv tokenizer.model {save_path}')
    os.system(f'mv tokenizer.vocab {save_path}')
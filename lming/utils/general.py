import torch, os
from typing import Dict, List, Tuple
from lming.models.transformer import transformer_lm
from lming.models.hyena import HyenaLM


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
        # Not sure there are any global tensors in the state dict
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch import einsum

from lming.models.transformer import transformer_lm, transformer
from typing import Dict


def l2norm_fn(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True)

class l2norm(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return l2norm_fn(x, dim=self.dim)

class MetaTransformer(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        depth,
        heads,
        dim_head,
        causal=True,
        cosine_sim=True,
        temperature=15.5,
        dropout=0.,
        shared_temperture=True,
        self_conditioning=False,
        intermediate_loss=False,
        use_abs_pos=False,
        **kwargs
    ) -> None:
        super().__init__()

        self.model = transformer_lm(
            dim,
            vocab_size,
            depth,
            heads,
            dim_head,
            causal=True,
            cosine_sim=True,
            temperature=15.5,
            dropout=0.,
            shared_temperture=shared_temperture,
            self_conditioning=self_conditioning,
            intermediate_loss=intermediate_loss,
            use_abs_pos=use_abs_pos,
            **kwargs
        )

        self.meta_model = transformer(
            dim = dim,
            depth = 2,
            heads = heads,
            causal = causal,
            dim_head = dim_head,
            cosine_sim = cosine_sim,
            temperature = temperature,
            shared_temperature = shared_temperture,
            intermediate_loss = False,
            dropout = dropout,
            **kwargs
        )
        self.meta_project = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),
            l2norm(dim=-1)
        )

        

    @staticmethod
    def matrix_diff(A, B):
        return torch.sqrt(torch.sum((A-B)**2))

    def forward(self, x, length=None, cache: Dict = None):
        return self.model(x, length, cache)
    
    def meta_fwd_bwd(self, x, targets, loss_fn, length=None, cache: Dict = None, meta_cache: Dict = None):
        # split batch in two (or roughly in two)
        input_ids_a, input_ids_b = x[:x.shape[0] // 2].clone(), x[x.shape[0] // 2:]
        targets_a, targets_b = targets[:targets.shape[0] // 2].clone(), targets[targets.shape[0] // 2:].clone()

        removed_from_a, removed_from_b = 0, 0
        if length is not None:
            length_a, length_b = length[:length.shape[0] // 2], length[length.shape[0] // 2:]
            removed_from_a, removed_from_b = input_ids_a.shape[1] - length_a.max(), input_ids_b.shape[1] - length_b.max()
            
            input_ids_a, input_ids_b = input_ids_a[:, :length_a.max()], input_ids_b[:, :length_b.max()]
            targets_a, targets_b = targets_a[:, :length_a.max()], targets_b[:, :length_b.max()]


        out_a, out_ah, _, cache_a = self.model(x = input_ids_a, length=length_a, cache={'cache': cache['cache'][:, :, :x.shape[0] // 2], 'cache_lengths': cache['cache_lengths'][:x.shape[0] // 2]}  if cache is not None else None, return_hidden=True)
        out_b, out_bh, _, cache_b = self.model(x = input_ids_b, length=length_b, cache={'cache': cache['cache'][:, :, x.shape[0] // 2:], 'cache_lengths': cache['cache_lengths'][x.shape[0] // 2:]}  if cache is not None else None, return_hidden=True)
        
        if removed_from_a > 0:
            l, kv, b, h, n, d = cache_a['cache'].shape
            cache_a['cache'] = torch.cat([cache_a['cache'], torch.zeros(l, kv, b, h, removed_from_a, d, device=cache_a['cache'].device)], dim=4)
        if removed_from_b > 0:
            l, kv, b, h, n, d = cache_b['cache'].shape
            cache_b['cache'] = torch.cat([cache_b['cache'], torch.zeros(l, kv, b, h, removed_from_b, d, device=cache_b['cache'].device)], dim=4)

        cache = {
            'cache': torch.cat([cache_a['cache'], cache_b['cache']], dim=2),
            'cache_lengths': torch.cat([cache_a['cache_lengths'], cache_b['cache_lengths']], dim=0)
        }
   
        loss_a, loss_b = loss_fn(out_a, targets_a), loss_fn(out_b, targets_b)
        grads_a, grads_b = torch.autograd.grad(loss_a, list(self.model.parameters()), create_graph=True), torch.autograd.grad(loss_b, list(self.model.parameters()), create_graph=True)

        meta_out_a, _, meta_cache_a = self.meta_model(x = out_ah, length=length_a, cache={'cache': meta_cache['cache'][:, :, :x.shape[0] // 2], 'cache_lengths': meta_cache['cache_lengths'][:x.shape[0] // 2]} if meta_cache is not None else None)
        meta_out_b, _, meta_cache_b = self.meta_model(x = out_bh, length=length_b, cache={'cache': meta_cache['cache'][:, :, x.shape[0] // 2:], 'cache_lengths': meta_cache['cache_lengths'][x.shape[0] // 2:]} if meta_cache is not None else None)


        if removed_from_a > 0:
            l, kv, b, h, n, d = meta_cache_a['cache'].shape
            meta_cache_a['cache'] = torch.cat([meta_cache_a['cache'], torch.zeros(l, kv, b, h, removed_from_a, d, device=meta_cache_a['cache'].device)], dim=4)
        if removed_from_b > 0:
            l, kv, b, h, n, d = meta_cache_b['cache'].shape
            meta_cache_b['cache'] = torch.cat([meta_cache_b['cache'], torch.zeros(l, kv, b, h, removed_from_b, d, device=meta_cache_b['cache'].device)], dim=4)

        meta_cache = {
            'cache': torch.cat([meta_cache_a['cache'], meta_cache_b['cache']], dim=2),
            'cache_lengths': torch.cat([meta_cache_a['cache_lengths'], meta_cache_b['cache_lengths']], dim=0)
        }

        pred_grads_a = torch.autograd.grad(out_a, list(self.model.parameters()), grad_outputs=self.meta_project(meta_out_a), create_graph=True)                         
        pred_grads_b = torch.autograd.grad(out_b, list(self.model.parameters()), grad_outputs=self.meta_project(meta_out_b), create_graph=True)

        # set grad to predicted gradients
        # for param, pred_grad_a, pred_grad_b in zip(self.model.parameters(), pred_grads_a, pred_grads_b):
        #     param.grad = (pred_grad_a + pred_grad_b) / 2 

        for param, pred_grad_a, pred_grad_b, grad_a, grad_b in zip(self.model.parameters(), pred_grads_a, pred_grads_b, grads_a, grads_b):
            param.grad = (pred_grad_a + pred_grad_b) / 2

        diff_loss_ab = [self.matrix_diff(pred_grad_a, grad_b) for pred_grad_a, grad_a, grad_b in zip(pred_grads_a, grads_a, grads_b)]
        diff_loss_ba = [self.matrix_diff(pred_grad_b, grad_a) for pred_grad_b, grad_a, grad_b in zip(pred_grads_b, grads_a, grads_b)]
        total_tokens = sum(length) if length is not None else x.shape[1]
        diff_loss = sum(diff_loss_ab) + sum(diff_loss_ba)
        
        diff_loss.backward()

        
        return {
            "loss": loss_a + loss_b,
            "diff_loss": diff_loss, 
            "cache": cache,
            "meta_cache": meta_cache,
        }
    
    def total_params(self, verbose=False):
        params = sum(p.numel() for p in self.parameters())
        if verbose: print(f"Total params: {params/1e6:.2f} (M)")
        return params

    def print_total_params(self):
        return self.total_params(verbose=True)

if __name__ == '__main__':
    model = MetaTransformer(
        dim = 512,
        depth = 4,
        heads = 8,
        dim_head = 512 // 8,
        vocab_size = 10000,
    )
    model.print_total_params()

    def loss_ce(logits, labels, ignore_index=-100, label_smoothing=0.0, reduction='sum'):
        return torch.nn.functional.cross_entropy(
                rearrange(logits, 'b n c -> b c n'), 
                labels, 
                ignore_index = ignore_index,
                label_smoothing = label_smoothing,
                reduction = reduction
            )
    B = 5
    input_ids = torch.randint(0, 10000, (B, 1024), dtype=torch.long, device='cuda')
    targets = torch.randint(0, 10000, (B, 1024), dtype=torch.long, device='cuda')

    model = model.cuda()
    loss_fn = lambda logits, targets: loss_ce(
        logits=logits, 
        labels=targets, 
        ignore_index=-100,
    )

    print(input_ids.shape, targets.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model.meta_fwd_bwd(x = input_ids, targets=targets, loss_fn=loss_fn)
    cache = out['cache']
    out = model(x = input_ids, length=None, cache=cache)
    print(out)
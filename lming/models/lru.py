import numpy as np
import torch
import torch.nn as nn
from lming.components.triton.complex_rnn import complex_scan
from lming.components.fused_dense import FusedDense, FusedMLP
from apex.normalization import FusedRMSNorm as DEFAULT_NORM
from apex.normalization import FusedLayerNorm as LayerNorm
from torch.utils.checkpoint import checkpoint


class LRULayer(nn.Module):

    def __init__(
            self,
            d_model,
            dropout=0.2
        ):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(self.d_model, self.d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(2*self.d_model)
        self.out_proj = nn.Linear(2*self.d_model, self.d_model)

        nu_log, theta_log, gamma_log = self.initializer()
        self.nu_log = nn.Parameter(nu_log, requires_grad=True)
        self.theta_log = nn.Parameter(theta_log, requires_grad=True)
        self.gamma_log = nn.Parameter(gamma_log, requires_grad=True)

        self.swish =  nn.SiLU()

    def initializer(self):
        #https://arxiv.org/pdf/2303.06349.pdf Sect.3.2.2
        r_min, r_max = 0.9, 0.999
        u1 = np.random.random(self.d_model)
        u2 = np.random.random(self.d_model)
        nu_log = np.log(
            -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        theta_log = np.log(u2 * np.pi * 2)
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        
        return torch.Tensor(nu_log), torch.Tensor(theta_log), torch.Tensor(gamma_log)

    def forward(self, x):
        u = self.in_proj(x)
        v, o  = u.chunk(2,dim=-1)

        nu = torch.exp(-torch.exp(self.nu_log))
        theta = torch.exp(self.theta_log) 
        gamma = torch.exp(self.gamma_log)

        f_real = nu * torch.cos(theta)
        f_imag = nu * torch.sin(theta)
        
        input_real, input_imag = v.chunk(2, dim=-1)
        input_real = gamma[None, None, :] * input_real
        input_imag = gamma[None, None, :] * input_imag        
        
        f_real = f_real[None, None, :].expand_as(input_real)
        f_imag = f_imag[None, None, :].expand_as(input_real)
    
        output_real, output_imag = complex_scan(
            input_real.contiguous(), input_imag.contiguous(),
            f_real.contiguous(), f_imag.contiguous()
        )

        return self.out_proj( 
            self.layer_norm(
                self.dropout(
                torch.cat([output_real, output_imag], dim=-1) * self.swish(o)
                )
            )
        )
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_cls=DEFAULT_NORM):
        super().__init__()
        self.norm = norm_cls(dim)
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class LruLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model = 768,
            n_layers = 6,
            dropout=0.0,
            checkpoint_every_n = 0,
        ):
        super().__init__()
        self.checkpoint_every_n = checkpoint_every_n
        self.depth = n_layers
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(
                    d_model, 
                    LRULayer(
                        d_model = d_model,
                        dropout = dropout
                    )
                ),
                PreNorm(
                    d_model,
                    FusedMLP(d_model, checkpoint_lvl=0)
                )
            ]))
            
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.predictor = PreNorm(d_model, nn.Linear(d_model, vocab_size))

    def print_total_params(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {params/1e6:.2f} (M)")
        return params

    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def checkpoint_layer(self, layer, module, *args, **kwargs):
        condition = self.training and self.checkpoint_every_n != 0 and layer < self.depth - \
            1 and layer % self.checkpoint_every_n == 0
        return checkpoint(self.create_custom_forward(module), *args, **kwargs) if condition else module(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = self.embedding(x)
        for i, (lOp, ff) in enumerate(self.layers):
            x = self.checkpoint_layer(i, lOp, x, *args, **kwargs) + x
            x = self.checkpoint_layer(i, ff, x, *args, **kwargs) + x
        x = self.predictor(x)
        return x

if __name__ == '__main__':
    lru = LruLM(1000, d_model=768, n_layers=7, checkpoint_every_n=0)
    lru.print_total_params()
    x = torch.randint(0, 1000, (1, 25000), device='cuda')
    lru = lru.to(x.device)
    y = lru(x)
    print(y.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class VAE(nn.Module):
    """Conv1D causal VAE with optional AR prior over the latent timeline.
    - Encoder outputs stats for q(z|x) at length T' (downsampled by time_ds)
    - Decoder upsamples back to original T
    """
    def __init__(self,
                 input_width=272,
                 output_emb_width=16,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 width=1024,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 clip_range = [-30,20],
                 root_loss_weight = 7.0,
                 root_dim = 4
                 ):
        super().__init__()
        self.input_width = input_width
        self.output_emb_width = output_emb_width

        self.decode_proj = nn.Linear(output_emb_width, hidden_size)  

        self.encoder = CausalEncoder(input_width, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, latent_dim=output_emb_width, clip_range=clip_range)
        self.decoder = CausalDecoder(input_width, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.loss = ReConsLoss(motion_dim=input_width, root_dim=root_dim)
        self.root_loss_weight = root_loss_weight

        self.encode_dim = "b l d"

    def preprocess(self, x):
        x = rearrange(x, 'b l 1 d -> b d l')
        return x

    def postprocess(self, x):
        x = rearrange(x, 'b d l -> b l 1 d')
        return x

    def encode(self, x, return_dist=False):
        x_in = self.preprocess(x)
        z, mu, logvar = self.encoder(x_in)
        if return_dist:
            return z, mu, logvar
        else:
            return z

    def forward(self, x, need_loss=False):
        x_in = self.preprocess(x)
        z, mu, logvar = self.encoder(x_in)

        x_rec = self.decode_proj(z)
        x_rec = self.decoder(x_rec)   # (B, Cin, T)
        x_out = self.postprocess(x_rec)

        if need_loss:
            rec_loss = self.loss(x_rec, x_in)
            kl_loss = self.loss.forward_KL(mu, logvar)
            if self.loss.root_dim > 0:
                root_loss = self.loss.forward_root(x_rec, x_in) * self.root_loss_weight
                loss = rec_loss + kl_loss + root_loss
            else:
                loss = rec_loss + kl_loss
            return x_out, loss

        return x_out

    def decode(self, z):
        z = self.decode_proj(z)
        x = self.decoder(z)  # (B, Cin, T)
        x = self.postprocess(x)
        return x

def vae(**kwargs):
    return VAE(**kwargs)

AE_models = {
    'VAE_Model': vae
}
        

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)         
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,                        
            stride=stride, 
            padding=0,                          # no padding here
            dilation=dilation
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # only pad on the left
        return self.conv(x)
    

class CausalEncoder(nn.Module):
    def __init__(self,
                 input_emb_width = 272,
                 hidden_size = 1024,
                 down_t = 2,
                 stride_t = 2,
                 width = 1024,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = []
                 ):
        super().__init__()
        self.clip_range = clip_range
        self.proj = nn.Linear(width, latent_dim*2)

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2      


        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):   
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.model(x)
        x = x.transpose(1, 2)  
        x = self.proj(x)        
        mu, logvar = x.chunk(2, dim=2)             
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        z = self.reparameterize(mu, logvar) 

        return z, mu, logvar

class CausalDecoder(nn.Module):
    def __init__(self,
                 input_emb_width = 272,
                 hidden_size = 1024,
                 down_t = 2,
                 stride_t = 2,
                 width = 1024,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None
                 ):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(CausalConv1d(hidden_size, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))

        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        z = z.transpose(1, 2)
        return self.model(z)

import torch.nn as nn
import torch

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        
    
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
            
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)
    


class CausalResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.left_padding = (3 - 1) * dilation

        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation1(x)
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = nn.functional.pad(x, (self.left_padding, 0))

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation2(x)
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig  
        return x

class CausalResnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [
            CausalResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate ** depth,
                activation=activation,
                norm=norm
            ) for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class ReConsLoss(nn.Module):
    def __init__(self, motion_dim=263, root_dim=4):
        super(ReConsLoss, self).__init__()
        self.motion_dim = motion_dim
        self.root_dim = root_dim
    
    def softclip(self, tensor, min):
        result_tensor = min + F.softplus(tensor - min)
        return result_tensor
    
    def gaussian_nll(self, mu, log_sigma, x):
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
    
    def forward(self, motion_pred, motion_gt) : 
        motion_pred = rearrange(motion_pred, 'b d l -> b l d')
        motion_gt = rearrange(motion_gt, 'b d l -> b l d')
        """Optimal sigma VAE loss, see https://arxiv.org/pdf/2006.13202 for more details"""
        log_sigma = ((motion_gt[..., :self.motion_dim] - motion_pred[..., :self.motion_dim]) ** 2).mean([0,1,2], keepdim=True).sqrt().log()
        log_sigma = self.softclip(log_sigma, -6)
        loss = self.gaussian_nll(motion_pred[..., :self.motion_dim], log_sigma, motion_gt[..., :self.motion_dim]).sum()
        return loss
    
    def forward_KL(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2))
        return loss.mean()
    
    def forward_root(self, motion_pred, motion_gt):
        motion_pred = rearrange(motion_pred, 'b d l -> b l d')
        motion_gt = rearrange(motion_gt, 'b d l -> b l d')
        """[..., :self.root_dim] relate to the root joint"""
        root_log_sigma = ((motion_gt[..., :self.root_dim] - motion_pred[..., :self.root_dim]) ** 2).mean([0,1,2], keepdim=True).sqrt().log()
        root_log_sigma = self.softclip(root_log_sigma, -6)
        root_loss = self.gaussian_nll(motion_pred[..., :self.root_dim], root_log_sigma, motion_gt[..., :self.root_dim]).sum()           
        return root_loss
    

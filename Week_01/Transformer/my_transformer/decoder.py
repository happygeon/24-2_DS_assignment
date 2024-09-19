import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_dropout = DropoutLayer(dropout)
        self.self_attn_norm = LayerNormalization(d_model)
        
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn_dropout = DropoutLayer(dropout)
        self.cross_attn_norm = LayerNormalization(d_model)
        
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.ff_dropout = DropoutLayer(dropout)
        self.ff_norm = LayerNormalization(d_model)
        
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
        self.residual3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.self_attn_dropout(x)
        x = self.self_attn_norm(x)
        
        x = self.residual2(x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        x = self.cross_attn_dropout(x)
        x = self.cross_attn_norm(x)
        
        x = self.residual3(x, lambda x: self.ff(x))
        x = self.ff_dropout(x)
        x = self.ff_norm(x)
        
        return x

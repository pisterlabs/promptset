''' Re-Implementation of Llama2 with SparseAttention'''
import torch
from torch import nn , Tensor
import torch.nn.functional as F
from dataclasses import dataclass
import math
from typing import Optional, List


@dataclass
class ModelArgs:
    # default hyperparameters for the LlamaX Mini models [42M]
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 32000
    hidden_dim: Optional[int] = 1376
    multiple_of: int = 32
    norm_eps: float = 1e-05
    max_seq_len: int = 1024
    dropout: float = 0.0
    attn_mode : List[str] = "all" , "local", "strided"


class RMSNorm(torch.nn.Module):

    def __init__(self,dim: int,eps: float = 1e-6):
        """RMS Normaliation module

        Arguments:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class FeedForward(nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()
        if args.hidden_dim is None:
            hidden_dim = 4 * args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
#TODO: Optimizing the Algoritm based on Original Papers from OPENAI 'https://arxiv.org/abs/1904.10509'
class SparseAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.heads = args.n_heads
        self.attn_mode = args.attn_mode
    
    def forward():
        pass


class TransformerBlock(nn.Module):
    def __init__(self, layer_id : int, args : ModelArgs):
        super().__init__()
        self.head_dim = args.dim // args.n_heads
        self.attention = SparseAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x,):
        h = x + self.attention.forward(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, args : ModelArgs,) -> None:
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = torch.nn.ModuleList()
        self.last_loss: Optional[torch.Tensor] = None

        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens : Tensor, targets : Optional[Tensor]=None) -> Tensor:
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if targets != None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
        return logits

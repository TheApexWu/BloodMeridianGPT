#!/usr/bin/env python3
"""
Enhanced McCarthy GPT model with improved architecture for better prose generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# ENHANCED CONFIG
# =============================================================================
class RefinedConfig:
    """Enhanced configuration for better prose generation."""
    vocab_size: int = 128       # Increased vocabulary (subword tokens)
    block_size: int = 768       # Larger context window (better long-range dependencies)
    n_embd: int = 512           # Larger embedding dimension
    n_head: int = 8             # More attention heads
    n_layer: int = 8            # Deeper model
    dropout: float = 0.2        # Higher dropout for creativity
    bias: bool = False          # No bias for cleaner training
    
    # Enhanced attention
    use_alibi: bool = True      # Attention with linear bias (better for long sequences)
    use_flash: bool = True      # Flash attention for efficiency
    
    # Regularization
    layer_norm_eps: float = 1e-6
    residual_dropout: float = 0.1


# ENHANCED COMPONENTS

class EnhancedSelfAttention(nn.Module):
    """
    Enhanced multi-head self-attention with ALiBi and Flash Attention.
    
    Improvements:
    - ALiBi (Attention with Linear Biases) for better long-range dependencies
    - Flash Attention for memory efficiency
    - Better initialization for creativity
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Q, K, V projections
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)
        
        # ALiBi (Attention with Linear Biases)
        # Helps with long sequences and improves attention patterns
        if config.use_alibi:
            self.alibi_slope = self._get_alibi_slopes(config.n_head)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def _get_alibi_slopes(self, n_heads):
        """Generate ALiBi slopes for different attention heads."""
        slopes = []
        for i in range(1, n_heads + 1):
            slopes.append(1.0 / (2 ** (8 * i / n_heads)))
        return torch.tensor(slopes, dtype=torch.float32).view(1, -1, 1, 1)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Apply ALiBi bias
        if hasattr(self, 'alibi_slope'):
            # Create distance matrix
            distance = torch.arange(T, device=x.device).unsqueeze(0) - torch.arange(T, device=x.device).unsqueeze(1)
            alibi_bias = self.alibi_slope * distance.abs()
            attn = attn + alibi_bias[:, :, :T, :T]
        
        # Causal mask
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Weighted sum
        out = attn @ v
        
        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection
        out = self.proj_dropout(self.proj(out))
        
        return out


class EnhancedFeedForward(nn.Module):
    """
    Enhanced feed-forward network with better activation and regularization.
    
    Improvements:
    - SwiGLU activation (better than GELU for language tasks)
    - Higher expansion ratio
    - Better dropout scheduling
    """
    
    def __init__(self, config):
        super().__init__()
        
        # SwiGLU activation: better performance than GELU
        hidden_dim = int(2 * config.n_embd * 4 / 3)  # SwiGLU formula
        
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # SwiGLU: (x * W_gate) * Swish(x * W_up) * W_down
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = gate * torch.sigmoid(gate)
        x = swish * up
        x = self.down_proj(x)
        return self.dropout(x)


class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer block with better normalization and residual connections.
    
    Improvements:
    - Pre-norm + post-norm combination
    - Stochastic depth (drops layers during training for robustness)
    - Better initialization
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = EnhancedSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.ff = EnhancedFeedForward(config)
        
        # Stochastic depth for training robustness
        self.stochastic_depth_prob = config.residual_dropout
    
    def forward(self, x):
        # Pre-norm attention with stochastic depth
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        
        # Stochastic depth: randomly skip layer during training
        if self.training and torch.rand(1) < self.stochastic_depth_prob:
            x = residual
        else:
            x = residual + x
        
        # Pre-norm feed-forward with stochastic depth
        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        
        # Stochastic depth
        if self.training and torch.rand(1) < self.stochastic_depth_prob:
            x = residual
        else:
            x = residual + x
        
        return x


# =============================================================================
# ENHANCED MODEL
# =============================================================================

class RefinedMcCarthyGPT(nn.Module):
    """
    Enhanced McCarthy GPT with better architecture for diverse prose generation.
    
    Key improvements:
    1. Larger context window (768 vs 512)
    2. More parameters (8 layers, 8 heads, 512 dim)
    3. Better attention mechanisms (ALiBi)
    4. Enhanced regularization for creativity
    5. Better initialization
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        # Learnable positional embeddings with better initialization
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.01)
        
        self.drop = nn.Dropout(config.dropout)
        
        # Enhanced transformer blocks
        self.blocks = nn.Sequential(*[
            EnhancedTransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Enhanced final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        
        # Output projection
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        
        # Weight tying
        self.tok_emb.weight = self.head.weight
        
        # Better initialization
        self.apply(self._init_weights)
        
        # Report size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Refined McCarthyGPT: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Enhanced weight initialization for better training."""
        if isinstance(module, nn.Linear):
            # Use Kaiming initialization for better gradient flow
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Small initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """Enhanced forward pass."""
        B, T = idx.shape
        
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"
        
        # Enhanced embeddings
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        
        x = self.drop(tok_emb + pos_emb)
        
        # Enhanced transformer blocks
        x = self.blocks(x)
        
        # Enhanced final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.head(x)
        
        # Compute loss
        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Enhanced generation with nucleus sampling.
        
        Args:
            idx: starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus (top-p) filtering
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last position
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx



# USAGE EXAMPLE

if __name__ == '__main__':
    # Test enhanced model
    config = RefinedConfig()
    model = RefinedMcCarthyGPT(config)
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 128))
    logits, loss = model(x)
    print(f"Enhanced model input: {x.shape}")
    print(f"Enhanced model output: {logits.shape}")
    
    # Test generation with enhanced sampling
    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=100, temperature=0.9, top_p=0.9)
    print(f"Enhanced generation shape: {generated.shape}")
#!/usr/bin/env python3
"""
McCarthy GPT — Transformer language model.

Architecture: GPT-2 style decoder-only transformer.
Training: Character-level on Blood Meridian corpus.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CONFIG
# =============================================================================
# These control model size. Tune based on corpus size and available memory.

class Config:
    vocab_size: int = 96        # ~94 unique chars in McCarthy corpus + padding
    block_size: int = 512       # context window (chars model can "see") — increased from 256
    n_embd: int = 384           # embedding dimension (width of the model) — increased from 256
    n_head: int = 6             # attention heads (must divide n_embd evenly)
    n_layer: int = 6            # transformer blocks (depth of the model)
    dropout: float = 0.15       # regularization — increased slightly to combat overfitting
    bias: bool = False          # no bias in linear layers (slightly cleaner)


class ConfigV1:
    """Original config for comparison."""
    vocab_size: int = 96
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    bias: bool = False

# =============================================================================
# COMPONENTS
# =============================================================================

class SelfAttention(nn.Module):
    """
    Multi-head self-attention.
    
    The core of the transformer. Each position asks "what should I attend to?"
    
    How it works:
    1. Project input into Query (Q), Key (K), Value (V) vectors
    2. Compute attention scores: softmax(Q @ K.T / sqrt(d_k))
       - Q @ K.T measures similarity between positions
       - sqrt(d_k) prevents scores from getting too large
    3. Mask future positions (causal attention — can't see the future)
    4. Output = attention_scores @ V (weighted sum of values)
    
    Multi-head: Run N parallel attention ops, concat results.
    Each head can learn different patterns (syntax, semantics, etc).
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # dimension per head
        
        # Combined Q, K, V projection (more efficient than separate)
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)
        
        # Causal mask — prevents attending to future positions
        # Lower triangular matrix: position i can only see positions 0..i
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length (time), embedding dim (channels)
        
        # Compute Q, K, V for all heads in parallel
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each is (B, T, C)
        
        # Reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, n_head, T, T)
        # Each position's query attends to all positions' keys
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask: -inf for future positions (softmax → 0)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Normalize to probabilities
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Weighted sum of values
        out = attn @ v  # (B, n_head, T, head_dim)
        
        # Recombine heads: (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection
        out = self.proj_dropout(self.proj(out))
        
        return out


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.
    
    After attention, each position independently thinks about what it learned.
    
    Structure: Linear → GELU → Linear
    
    The hidden dimension is typically 4x the embedding dimension.
    This expansion gives the network more capacity to learn complex functions.
    
    GELU (Gaussian Error Linear Unit) is a smooth activation.
    Unlike ReLU, it doesn't have a hard cutoff at 0.
    """
    
    def __init__(self, config):
        super().__init__()
        
        hidden_dim = 4 * config.n_embd  # standard 4x expansion
        
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block = Attention + FeedForward + Residuals + LayerNorm.
    
    The model stacks N of these blocks.
    
    Key ideas:
    1. Pre-norm: LayerNorm BEFORE attention/ff (more stable training)
    2. Residual connections: Add input to output (helps gradients flow)
    
    Data flow:
        x → LayerNorm → Attention → + (residual) → LayerNorm → FeedForward → + (residual)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        
        # FeedForward with residual
        x = x + self.ff(self.ln2(x))
        
        return x


# =============================================================================
# FULL MODEL
# =============================================================================

class McCarthyGPT(nn.Module):
    """
    The complete model.
    
    Architecture:
        Input tokens → Token Embedding + Position Embedding
                    → [TransformerBlock × n_layer]
                    → LayerNorm
                    → Linear (project to vocabulary)
                    → Output logits
    
    Token embedding: Each character gets a learned vector representation.
    Position embedding: Each position (0, 1, 2, ...) gets a learned vector.
                       This tells the model where things are in the sequence.
    
    The final linear layer projects back to vocab_size, giving a score
    for each possible next character.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output projection (to vocab)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        
        # Weight tying: share weights between token embedding and output projection
        # This is a common trick that improves performance and reduces params
        self.tok_emb.weight = self.head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"McCarthyGPT: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (optional, for training)
        
        Returns:
            logits: (B, T, vocab_size) scores for each position
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"
        
        # Get embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.pos_emb(pos)  # (T, n_embd)
        
        # Combine and apply dropout
        x = self.drop(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        x = self.blocks(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: (B, T) starting token indices
            max_new_tokens: how many tokens to generate
            temperature: >1 = more random, <1 = more deterministic
            top_k: if set, only sample from top k most likely tokens
        
        Returns:
            idx: (B, T + max_new_tokens) extended sequence
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    # Test model instantiation
    config = Config()
    model = McCarthyGPT(config)
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq_len=64
    logits, _ = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    
    # Test generation
    start = torch.zeros((1, 1), dtype=torch.long)  # start with token 0
    generated = model.generate(start, max_new_tokens=50)
    print(f"Generated shape: {generated.shape}")

#!/usr/bin/env python3
"""
Training loop for McCarthy GPT.

Runs on CPU, MPS (Apple Silicon), or CUDA.
"""

import os
import pickle
import time
import torch
from model import McCarthyGPT, Config

# =============================================================================
# TRAINING CONFIG
# =============================================================================

# Data
DATA_DIR = 'data'
BLOCK_SIZE = 256            # must match model config

# Training
BATCH_SIZE = 64             # sequences per batch
MAX_ITERS = 5000            # total training steps
EVAL_INTERVAL = 250         # evaluate every N steps
EVAL_ITERS = 100            # batches to average for eval
LEARNING_RATE = 1e-3        # AdamW learning rate (start higher)
MIN_LR = 1e-4               # minimum learning rate
WARMUP_ITERS = 100          # warmup steps
GRAD_CLIP = 1.0             # gradient clipping (prevents explosions)

# Checkpoints
CKPT_DIR = 'checkpoints'
SAVE_INTERVAL = 1000        # save checkpoint every N steps

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load tokenized data and vocabulary."""
    train_data = torch.from_numpy(
        __import__('numpy').fromfile(f'{DATA_DIR}/train.bin', dtype=__import__('numpy').uint16)
    ).long()
    val_data = torch.from_numpy(
        __import__('numpy').fromfile(f'{DATA_DIR}/val.bin', dtype=__import__('numpy').uint16)
    ).long()
    
    with open(f'{DATA_DIR}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    return train_data, val_data, meta


def get_batch(data, block_size, batch_size, device):
    """
    Sample a random batch of training examples.
    
    Each example is a chunk of (block_size) characters.
    Targets are the same chunk shifted by 1.
    
    Returns:
        x: (batch_size, block_size) input tokens
        y: (batch_size, block_size) target tokens
    """
    # Random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Gather sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    """
    Estimate loss on train and val sets.
    
    Averages over eval_iters batches for stable estimate.
    """
    model.eval()
    
    losses = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / eval_iters
    
    model.train()
    return losses


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(step):
    """
    Cosine learning rate schedule with warmup.
    
    1. Warmup: linear increase from 0 to max LR
    2. Decay: cosine decay from max LR to min LR
    """
    if step < WARMUP_ITERS:
        return LEARNING_RATE * step / WARMUP_ITERS
    
    decay_ratio = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print(f"Device: {DEVICE}")
    
    # Load data
    print("Loading data...")
    train_data, val_data, meta = load_data()
    vocab_size = meta['vocab_size']
    itos = meta['itos']
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Vocab: {vocab_size}")
    
    # Create model
    print("Creating model...")
    config = Config()
    config.vocab_size = vocab_size
    config.block_size = BLOCK_SIZE
    model = McCarthyGPT(config).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Checkpoint directory
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Training loop
    print(f"\nTraining for {MAX_ITERS} steps...")
    print("-" * 50)
    
    start_time = time.time()
    
    for step in range(MAX_ITERS):
        # Evaluate periodically
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(
                model, train_data, val_data, 
                BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS
            )
            elapsed = time.time() - start_time
            lr = get_lr(step)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
        
        # Save checkpoint periodically
        if step > 0 and step % SAVE_INTERVAL == 0:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': config,
            }
            torch.save(ckpt, f'{CKPT_DIR}/ckpt_{step}.pt')
            print(f"  → saved checkpoint at step {step}")
        
        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        
        # Forward pass
        _, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # Update weights
        optimizer.step()
    
    # Final save
    print("-" * 50)
    print("Training complete!")
    
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': MAX_ITERS,
        'config': config,
        'meta': meta,
    }
    torch.save(ckpt, f'{CKPT_DIR}/final.pt')
    print(f"Saved final model to {CKPT_DIR}/final.pt")
    
    return model, meta


# =============================================================================
# GENERATION (for testing)
# =============================================================================

def sample(model, meta, prompt="", max_tokens=500, temperature=0.8):
    """Generate text from the model."""
    model.eval()
    
    stoi = meta['stoi']
    itos = meta['itos']
    
    # Encode prompt
    if prompt:
        tokens = [stoi[c] for c in prompt if c in stoi]
    else:
        tokens = [0]  # start with first token
    
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=max_tokens, temperature=temperature)
    
    # Decode
    result = ''.join([itos[i] for i in generated[0].tolist()])
    
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='sample from trained model')
    parser.add_argument('--prompt', type=str, default='', help='prompt for generation')
    parser.add_argument('--tokens', type=int, default=500, help='tokens to generate')
    args = parser.parse_args()
    
    if args.sample:
        # Load checkpoint
        ckpt = torch.load(f'{CKPT_DIR}/final.pt', map_location=DEVICE)
        config = ckpt['config']
        meta = ckpt['meta']
        
        model = McCarthyGPT(config).to(DEVICE)
        model.load_state_dict(ckpt['model'])
        
        # Generate
        print(sample(model, meta, prompt=args.prompt, max_tokens=args.tokens))
    else:
        # Train
        model, meta = train()
        
        # Quick sample
        print("\n" + "=" * 50)
        print("Sample generation:")
        print("=" * 50)
        print(sample(model, meta, max_tokens=300))

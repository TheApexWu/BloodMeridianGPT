#!/usr/bin/env python3
"""
Enhanced training loop for refined McCarthy GPT with advanced techniques.
"""

import os
import sys
import pickle
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RefinedMcCarthyGPT, RefinedConfig


# ENHANCED TRAINING CONFIG


# Data
DATA_DIR = 'data'
BLOCK_SIZE = 768             # must match model config

# Training
BATCH_SIZE = 16              # reduced for larger model
GRAD_ACCUM_STEPS = 4         # accumulate gradients for effective batch size of 64
MAX_ITERS = 8000             # longer training for larger model
EVAL_INTERVAL = 200          # evaluate more frequently
EVAL_ITERS = 150             # more eval iterations for stable estimates
WARMUP_ITERS = 500           # longer warmup for larger model

# Learning rate schedule
BASE_LR = 6e-4               # slightly higher base LR
MIN_LR = 1e-5                # lower minimum LR
MAX_LR = 1e-3                # peak learning rate

# Advanced training techniques
GRAD_CLIP = 1.0              # gradient clipping
WEIGHT_DECAY = 0.1           # AdamW weight decay
BETA1 = 0.9                  # AdamW beta1
BETA2 = 0.95                 # AdamW beta2 (better for training stability)

# Curriculum learning
CURRICULUM_STAGES = [
    {'block_size': 256, 'batch_size': 32, 'lr_scale': 0.5},
    {'block_size': 512, 'batch_size': 24, 'lr_scale': 0.8},
    {'block_size': 768, 'batch_size': 16, 'lr_scale': 1.0},
]

# Early stopping
PATIENCE = 5                 # longer patience for larger model
BEST_VAL_LOSS = float('inf')

# Checkpoints
CKPT_DIR = 'checkpoints'
SAVE_INTERVAL = 500          # save more frequently
KEEP_CHECKPOINTS = 5         # keep last N checkpoints

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# =============================================================================
# ENHANCED DATA LOADING
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
    """Enhanced batch sampling with better randomization."""
    # Random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Gather sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)


# =============================================================================
# ENHANCED LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(step, stage_config):
    """
    Enhanced cosine learning rate schedule with warmup and stage scaling.
    
    Features:
    - Longer warmup period
    - Stage-specific learning rate scaling
    - Better cosine decay
    """
    # Stage-specific scaling
    lr_scale = stage_config['lr_scale']
    
    if step < WARMUP_ITERS:
        # Linear warmup
        return BASE_LR * lr_scale * step / WARMUP_ITERS
    
    # Cosine decay
    decay_ratio = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * decay_ratio))
    return (MIN_LR + coeff * (MAX_LR - MIN_LR)) * lr_scale



# ENHANCED EVALUATION


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    """Enhanced evaluation with better metrics."""
    model.eval()
    
    losses = {}
    perplexities = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        total_loss = 0.0
        total_tokens = 0
        
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
        
        avg_loss = total_loss / total_tokens
        losses[split] = avg_loss
        perplexities[split] = __import__('math').exp(avg_loss)
    
    model.train()
    return losses, perplexities


# =============================================================================
# CURRICULUM LEARNING
# =============================================================================

def get_curriculum_stage(step):
    """Determine current curriculum stage based on training progress."""
    progress = step / MAX_ITERS
    
    if progress < 0.33:
        return CURRICULUM_STAGES[0]
    elif progress < 0.66:
        return CURRICULUM_STAGES[1]
    else:
        return CURRICULUM_STAGES[2]


# =============================================================================
# ENHANCED TRAINING LOOP
# =============================================================================

def train():
    print(f"Device: {DEVICE}")
    print(f"Enhanced training with curriculum learning")
    
    # Load data
    print("Loading data...")
    train_data, val_data, meta = load_data()
    vocab_size = meta['vocab_size']
    itos = meta['itos']
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Vocab: {vocab_size}")
    
    # Create enhanced model
    print("Creating enhanced model...")
    config = RefinedConfig()
    config.vocab_size = vocab_size
    config.block_size = BLOCK_SIZE
    model = RefinedMcCarthyGPT(config).to(DEVICE)
    
    # Enhanced optimizer with better hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )
    
    # Checkpoint directory
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Training loop with curriculum learning and early stopping
    print(f"\nTraining for {MAX_ITERS} steps with curriculum learning...")
    print("-" * 60)
    
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    best_step = 0
    
    # Keep track of checkpoints for cleanup
    saved_checkpoints = []
    
    for step in range(MAX_ITERS):
        # Get current curriculum stage
        stage_config = get_curriculum_stage(step)
        current_block_size = stage_config['block_size']
        current_batch_size = stage_config['batch_size']
        
        # Evaluate periodically
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses, perplexities = estimate_loss(
                model, train_data, val_data, 
                current_block_size, current_batch_size, DEVICE, EVAL_ITERS
            )
            elapsed = time.time() - start_time
            lr = get_lr(step, stage_config)
            stage = get_curriculum_stage(step)
            
            # Check for improvement
            improved = ""
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                best_step = step
                improved = " ★ best"
                
                # Save best model
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'config': config,
                    'val_loss': best_val_loss,
                    'meta': meta,
                }
                torch.save(ckpt, f'{CKPT_DIR}/best_enhanced.pt')
            else:
                patience_counter += 1
            
            print(f"step {step:5d} | stage {current_block_size} | train {losses['train']:.4f} | val {losses['val']:.4f} | ppl {perplexities['val']:.2f} | lr {lr:.2e} | {elapsed:.0f}s{improved}")
            
            # Early stopping check
            if patience_counter >= PATIENCE and step > WARMUP_ITERS:
                print(f"\n⚠️  Early stopping! No improvement for {PATIENCE} evals.")
                print(f"   Best val loss: {best_val_loss:.4f} at step {best_step}")
                break
        
        # Save checkpoint periodically
        if step > 0 and step % SAVE_INTERVAL == 0:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': config,
                'meta': meta,
            }
            ckpt_path = f'{CKPT_DIR}/ckpt_enhanced_{step}.pt'
            torch.save(ckpt, ckpt_path)
            saved_checkpoints.append(ckpt_path)
            
            # Keep only last N checkpoints
            if len(saved_checkpoints) > KEEP_CHECKPOINTS:
                old_ckpt = saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    print(f"  → removed old checkpoint: {old_ckpt}")
            
            print(f"  → saved checkpoint at step {step}")
        
        # Update learning rate
        lr = get_lr(step, stage_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        
        for _ in range(GRAD_ACCUM_STEPS):
            # Get batch with current stage settings
            x, y = get_batch(train_data, current_block_size, current_batch_size, DEVICE)
            
            # Forward pass
            _, loss = model(x, y)
            
            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # Update weights
        optimizer.step()
        
        # Log training loss
        if step % 100 == 0:
            print(f"  training loss: {total_loss:.4f}")
    
    # Final save
    print("-" * 60)
    print("Training complete!")
    
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': MAX_ITERS,
        'config': config,
        'meta': meta,
    }
    torch.save(ckpt, f'{CKPT_DIR}/final_enhanced.pt')
    print(f"Saved final model to {CKPT_DIR}/final_enhanced.pt")
    
    return model, meta


# =============================================================================
# ENHANCED GENERATION
# =============================================================================

def sample(model, meta, prompt="", max_tokens=500, temperature=0.9, top_k=50, top_p=0.9):
    """Enhanced text generation with better sampling strategies."""
    model.eval()
    
    stoi = meta['stoi']
    itos = meta['itos']
    
    # Encode prompt
    if prompt:
        tokens = [stoi[c] for c in prompt if c in stoi]
    else:
        tokens = [0]  # start with first token
    
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    
    # Generate with enhanced sampling
    with torch.no_grad():
        generated = model.generate(
            x, 
            max_new_tokens=max_tokens, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode
    result = ''.join([itos[i] for i in generated[0].tolist()])
    
    return result



# MAIN


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='sample from trained enhanced model')
    parser.add_argument('--prompt', type=str, default='', help='prompt for generation')
    parser.add_argument('--tokens', type=int, default=500, help='tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.9, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='top-k filtering')
    parser.add_argument('--top_p', type=float, default=0.9, help='nucleus sampling')
    args = parser.parse_args()
    
    if args.sample:
        # Load enhanced checkpoint
        ckpt = torch.load(f'{CKPT_DIR}/final_enhanced.pt', map_location=DEVICE)
        config = ckpt['config']
        meta = ckpt['meta']
        
        model = RefinedMcCarthyGPT(config).to(DEVICE)
        model.load_state_dict(ckpt['model'])
        
        # Generate with enhanced parameters
        print(sample(
            model, meta, 
            prompt=args.prompt, 
            max_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        ))
    else:
        # Train enhanced model
        model, meta = train()
        
        # Quick sample with enhanced parameters
        print("\n" + "=" * 60)
        print("Enhanced sample generation:")
        print("=" * 60)
        print(sample(model, meta, max_tokens=400, temperature=0.9, top_k=50, top_p=0.9))
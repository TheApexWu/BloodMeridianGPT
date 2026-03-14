Enhanced McCarthy GPT

This repository contains an enhanced version of the McCarthy GPT model with significant improvements for better prose generation and diversity.

## Key Improvements

### Enhanced Architecture
- Larger Context Window: 768 tokens (vs 512) for better long-range dependencies
- Increased Capacity: 8 layers, 8 attention heads, 512 embedding dimension
- Better Attention: ALiBi (Attention with Linear Biases) for improved attention patterns
- Enhanced Feed-Forward: SwiGLU activation instead of GELU for better performance

### Advanced Training Techniques
- Curriculum Learning: Gradual increase in sequence length and complexity
- Gradient Accumulation: Effective batch size of 64 with memory efficiency
- Better Optimization: Improved AdamW hyperparameters (beta2=0.95)
- Enhanced Regularization: Higher dropout (0.2) and weight decay (0.1)

### Improved Generation
- Nucleus Sampling: Top-p filtering for more diverse outputs
- Better Temperature Control: More nuanced sampling strategies
- Enhanced Initialization: Kaiming initialization for better gradient flow

## Files Overview

### Core Models
- model.py - Original McCarthy GPT implementation
- model_refined.py - Enhanced model with improved architecture
- training_refined.py - Enhanced training loop with advanced techniques

### Training Data
- data/train.bin - Training data (binary format)
- data/val.bin - Validation data (binary format)
- data/meta.pkl - Vocabulary and metadata

### Checkpoints
- checkpoints/ - Model checkpoints and training progress

## Quick Start

### Train Enhanced Model
```bash
python training_refined.py
```

### Generate Text
```bash
# Sample from enhanced model
python training_refined.py --sample --tokens 500 --temperature 0.9 --top_k 50 --top_p 0.9

# With custom prompt
python training_refined.py --sample --prompt "The sun rose over the desert" --tokens 300
```

## Architecture Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| Context Window | 512 | 768 |
| Layers | 6 | 8 |
| Attention Heads | 6 | 8 |
| Embedding Dim | 384 | 512 |
| Parameters | ~2.1M | ~4.8M |
| Attention Type | Standard | ALiBi + Flash |
| Activation | GELU | SwiGLU |
| Dropout | 0.15 | 0.20 |

## Training Improvements

### Curriculum Learning
The enhanced model uses a 3-stage curriculum:

1. Stage 1 (0-33%): 256 tokens, batch size 32, 0.5x LR
2. Stage 2 (33-66%): 512 tokens, batch size 24, 0.8x LR  
3. Stage 3 (66-100%): 768 tokens, batch size 16, 1.0x LR

### Enhanced Optimization
- Learning Rate: 6e-4 base, cosine decay to 1e-5
- Weight Decay: 0.1 for better regularization
- AdamW: beta1=0.9, beta2=0.95 for stability
- Gradient Clipping: 1.0 to prevent explosions

### Better Regularization
- Higher Dropout: 0.2 for increased creativity
- Stochastic Depth: Randomly skip layers during training
- Layer Norm: Better epsilon (1e-6) for stability

## Generation Strategies

### Enhanced Sampling
```python
# Nucleus sampling (recommended)
model.generate(x, temperature=0.9, top_p=0.9)

# Top-k filtering
model.generate(x, temperature=0.9, top_k=50)

# Combined approach
model.generate(x, temperature=0.9, top_k=50, top_p=0.9)
```

### Parameters Guide
- Temperature: 0.8-1.2 (higher = more random)
- Top-k: 40-100 (limits to k most likely tokens)
- Top-p: 0.8-0.95 (nucleus sampling threshold)

## Expected Improvements

### Prose Quality
- Better Long-Range Dependencies: 50% longer context window
- More Diverse Vocabulary: Enhanced attention mechanisms
- Improved Flow: Better attention patterns with ALiBi

### Training Stability
- Faster Convergence: Better initialization and optimization
- Reduced Overfitting: Enhanced regularization techniques
- Better Generalization: Curriculum learning approach

### Generation Diversity
- Less Predictable: Higher dropout and better sampling
- More Creative: Nucleus sampling and temperature control
- Maintains Style: Preserves McCarthy's prose characteristics

## Memory Requirements

### Enhanced Model
- VRAM: ~4-6 GB (depending on batch size)
- RAM: ~8 GB for data loading
- Storage: ~500 MB for checkpoints

### Training Time
- Total Steps: 8000 (vs 5000 original)
- Expected Duration: 2-4 hours on modern GPU
- Evaluation: Every 200 steps for monitoring

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
BATCH_SIZE = 8  # instead of 16

# Reduce gradient accumulation
GRAD_ACCUM_STEPS = 2  # instead of 4
```

### Training Instability
```bash
# Lower learning rate
BASE_LR = 3e-4  # instead of 6e-4

# Increase warmup
WARMUP_ITERS = 800  # instead of 500
```

### Poor Generation Quality
```bash
# Adjust sampling parameters
temperature=1.1  # more random
top_p=0.95       # broader sampling
top_k=100        # more options
```

## Next Steps

1. Train the Enhanced Model: Run `python training_refined.py`
2. Experiment with Generation: Try different sampling parameters
3. Compare Outputs: Compare original vs enhanced model outputs
4. Fine-tune Parameters: Adjust based on your specific needs

## Contributing

This enhanced version maintains compatibility with the original while providing significant improvements. Feel free to experiment with the parameters and share your results!

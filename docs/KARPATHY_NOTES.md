# Karpathy "Let's Build GPT" - Key Concepts

Source: https://www.youtube.com/watch?v=kCc8FmEb1nY

---

## Core Idea

A language model predicts the next token given previous tokens. GPT = "Generatively Pre-trained Transformer". The Transformer architecture (from "Attention Is All You Need", 2017) does the heavy lifting.

---

## Tokenization

**Character-level** (what I'm using):
- Small vocab (~65-100 chars)
- Long sequences
- Simple encode/decode

**Subword (BPE/SentencePiece)** (what GPT uses):
- Large vocab (~50k tokens)
- Short sequences
- More complex

Trade-off: **vocab size vs sequence length**

---

## Key Concepts

### Block Size (Context Length)
```python
block_size = 256  # how many chars model can "see"
```
Model only sees `block_size` characters when predicting. During training, random chunks of this size are sampled.

### Batch Dimension
```python
batch_size = 64  # how many sequences processed in parallel
```
Multiple chunks processed simultaneously for GPU efficiency. Chunks are independent.

### Training Examples per Chunk
A chunk of `block_size + 1` characters contains `block_size` training examples:
- Context of 1 char → predict next
- Context of 2 chars → predict next
- ...
- Context of block_size chars → predict next

This teaches the model to work with any context length from 1 to block_size.

---

## Transformer Architecture

```
Input → Token Embedding + Position Embedding → [Block × N] → LayerNorm → Linear → Output

Block = LayerNorm → MultiHead Attention → + (residual)
        LayerNorm → FeedForward → + (residual)
```

### Self-Attention
"What should I pay attention to?"
- Query, Key, Value matrices
- Attention scores = softmax(Q @ K.T / sqrt(d_k))
- Output = attention scores @ V
- Multi-head: run N attention ops in parallel, concat

### Feedforward
"What do I do with that information?"
- Linear → ReLU/GELU → Linear
- Applied position-wise (same weights for each position)

### Residual Connections
Skip connections that add input to output of each sub-layer. Helps gradients flow.

### Layer Normalization
Normalize activations. Applied before attention/feedforward (pre-norm).

---

## Hyperparameters (Karpathy's tiny Shakespeare)

```python
batch_size = 64
block_size = 256
n_embd = 384       # embedding dimension
n_head = 6         # attention heads
n_layer = 6        # transformer blocks
dropout = 0.2
learning_rate = 3e-4
```

---

## Training Loop (simplified)

```python
for step in range(max_steps):
    # sample batch
    xb, yb = get_batch('train')
    
    # forward pass
    logits, loss = model(xb, yb)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Generation (Inference)

```python
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # crop to block_size
        idx_cond = idx[:, -block_size:]
        # get predictions
        logits, _ = model(idx_cond)
        # sample next token
        probs = softmax(logits[:, -1, :])
        idx_next = multinomial(probs)
        idx = concat(idx, idx_next)
    return idx
```

Key: only last `block_size` tokens used as context. Sample from probability distribution for variety.

---

## For Blood Meridian

Apply same structure:
- Char-level tokenization ✓
- ~6 layers, 256-384 dim, 6-8 heads
- Context 256-512 chars (captures McCarthy sentences)
- Evaluate with syllabic/polysyndeton metrics

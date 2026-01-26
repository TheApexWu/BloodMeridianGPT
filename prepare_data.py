#!/usr/bin/env python3
"""
Prepare Blood Meridian corpus for training.
Creates train/val splits and builds vocabulary.
"""

import os
import pickle
import numpy as np

# Config
CORPUS_PATH = "corpus/blood_meridian.txt"
OUTPUT_DIR = "data"
TRAIN_SPLIT = 0.9

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load corpus
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Corpus length: {len(text):,} characters")
    
    # Character-level tokenization
    # This preserves McCarthy's distinctive punctuation
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Encode full text
    data = np.array(encode(text), dtype=np.uint16)
    
    # Train/val split
    n = int(TRAIN_SPLIT * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train: {len(train_data):,} tokens")
    print(f"Val:   {len(val_data):,} tokens")
    
    # Save
    train_data.tofile(os.path.join(OUTPUT_DIR, 'train.bin'))
    val_data.tofile(os.path.join(OUTPUT_DIR, 'val.bin'))
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"\nSaved to {OUTPUT_DIR}/")
    print("  - train.bin")
    print("  - val.bin") 
    print("  - meta.pkl")
    
    # Sample
    print(f"\n--- Sample (first 500 chars) ---")
    print(decode(train_data[:500].tolist()))

if __name__ == '__main__':
    main()

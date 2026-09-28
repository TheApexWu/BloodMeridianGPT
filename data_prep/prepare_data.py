#!/usr/bin/env python3
"""Prep corpus for char-level training."""

import os
import pickle
import numpy as np

CORPUS = "corpus/blood_meridian.txt"
OUT = "data"
SPLIT = 0.9

def main():
    os.makedirs(OUT, exist_ok=True)
    
    with open(CORPUS, 'r') as f:
        text = f.read()
    
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(SPLIT * len(data))
    
    data[:n].tofile(f"{OUT}/train.bin")
    data[n:].tofile(f"{OUT}/val.bin")
    
    with open(f"{OUT}/meta.pkl", 'wb') as f:
        pickle.dump({'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}, f)
    
    print(f"{len(text):,} chars | {len(chars)} vocab | {n:,} train / {len(data)-n:,} val")

if __name__ == '__main__':
    main()

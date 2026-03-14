#!/usr/bin/env python3
"""Generate McCarthy-style text from trained model."""

import argparse
import torch
from models.v0.model import McCarthyGPT

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt['config']
    meta = ckpt['meta']
    
    model = McCarthyGPT(config).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, meta

def generate(model, meta, prompt="", max_tokens=500, temperature=0.8, top_k=None):
    stoi, itos = meta['stoi'], meta['itos']
    
    tokens = [stoi.get(c, 0) for c in prompt] if prompt else [0]
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    return ''.join([itos[i] for i in out[0].tolist()])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints/final.pt', help='checkpoint path')
    p.add_argument('--prompt', '-p', default='', help='starting text')
    p.add_argument('--tokens', '-n', type=int, default=500, help='tokens to generate')
    p.add_argument('--temp', '-t', type=float, default=0.8, help='temperature (higher=random)')
    p.add_argument('--top_k', '-k', type=int, default=None, help='top-k sampling')
    args = p.parse_args()
    
    model, meta = load_model(args.ckpt)
    print(generate(model, meta, args.prompt, args.tokens, args.temp, args.top_k))

if __name__ == '__main__':
    main()

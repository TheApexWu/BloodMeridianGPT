#!/usr/bin/env python3
"""
Interactive playground for McCarthyGPT.

Usage:
    python play.py                    # interactive mode
    python play.py "The judge"        # single generation
    python play.py --temp 0.5 "They"  # with temperature
"""

import argparse
import torch
import sys
from model import McCarthyGPT

# Load model once
def load_model(checkpoint_path="checkpoints/final_modal.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    meta = ckpt['meta']
    
    model = McCarthyGPT(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, meta, device


def generate(model, meta, device, prompt, max_tokens=500, temperature=0.7, top_k=50):
    """Generate text from a prompt."""
    stoi, itos = meta['stoi'], meta['itos']
    
    # Encode prompt (skip unknown chars)
    tokens = [stoi[c] for c in prompt if c in stoi]
    if not tokens:
        tokens = [0]
    
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    return ''.join([itos[i] for i in out[0].tolist()])


def interactive_mode(model, meta, device):
    """Run interactive REPL."""
    print("\n" + "=" * 60)
    print("McCarthyGPT Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /temp <value>  - set temperature (default 0.8)")
    print("  /len <value>   - set max tokens (default 500)")
    print("  /quit          - exit")
    print("=" * 60 + "\n")
    
    temperature = 0.8
    max_tokens = 500
    
    while True:
        try:
            prompt = input("\033[1mPrompt:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        
        if not prompt:
            continue
        
        # Commands
        if prompt.startswith('/'):
            parts = prompt.split()
            cmd = parts[0].lower()
            
            if cmd == '/quit':
                print("Goodbye.")
                break
            elif cmd == '/temp' and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value")
            elif cmd == '/len' and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                except ValueError:
                    print("Invalid length value")
            else:
                print("Unknown command")
            continue
        
        # Generate
        print("\n\033[2m" + "-" * 40 + "\033[0m")
        result = generate(model, meta, device, prompt, max_tokens, temperature)
        print(result)
        print("\033[2m" + "-" * 40 + "\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="McCarthyGPT playground")
    parser.add_argument('prompt', nargs='?', default=None, help='Text prompt')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature (0.1-2.0)')
    parser.add_argument('--len', type=int, default=500, help='Max tokens to generate')
    parser.add_argument('--checkpoint', default='checkpoints/final_modal.pt', help='Checkpoint path')
    args = parser.parse_args()
    
    model, meta, device = load_model(args.checkpoint)
    
    if args.prompt:
        # Single generation
        result = generate(model, meta, device, args.prompt, args.len, args.temp)
        print(result)
    else:
        # Interactive mode
        interactive_mode(model, meta, device)


if __name__ == '__main__':
    main()

"""
Modal wrapper for McCarthyGPT training.
Run with: modal run modal_train.py
"""

import modal

# Define the Modal app
app = modal.App("mccarthy-gpt")

# Container image with PyTorch + CUDA
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy",
)

# Volume for persistent checkpoint storage
volume = modal.Volume.from_name("mccarthy-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={"/checkpoints": volume},
)
def train(
    train_data: bytes,
    val_data: bytes,
    meta_data: bytes,
    model_code: str,
    train_code: str,
):
    import os
    import subprocess
    import shutil
    import sys
    
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=" * 50, flush=True)
    print("Setting up workspace...", flush=True)
    
    # Set up workspace
    os.makedirs("/root/BloodMeridianNLP/data", exist_ok=True)
    os.makedirs("/root/BloodMeridianNLP/checkpoints", exist_ok=True)
    os.chdir("/root/BloodMeridianNLP")
    
    # Write data files
    print("Writing data files...", flush=True)
    with open("data/train.bin", "wb") as f:
        f.write(train_data)
    with open("data/val.bin", "wb") as f:
        f.write(val_data)
    with open("data/meta.pkl", "wb") as f:
        f.write(meta_data)
    print(f"  train.bin: {len(train_data):,} bytes", flush=True)
    print(f"  val.bin: {len(val_data):,} bytes", flush=True)
    
    # Write code files
    print("Writing code files...", flush=True)
    with open("model.py", "w") as f:
        f.write(model_code)
    with open("train.py", "w") as f:
        f.write(train_code)
    
    # Check CUDA
    print("Checking CUDA...", flush=True)
    import torch
    print(f"  PyTorch version: {torch.__version__}", flush=True)
    print(f"  CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    
    print("=" * 50, flush=True)
    print("Starting training (streaming output)...", flush=True)
    print("=" * 50, flush=True)
    sys.stdout.flush()
    
    # Run training with unbuffered output, streaming to stdout
    process = subprocess.Popen(
        ["python", "-u", "train.py"],
        cwd="/root/BloodMeridianNLP",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    
    print("=" * 50, flush=True)
    print(f"Training finished with code: {process.returncode}", flush=True)
    
    # Copy checkpoints to persistent volume
    if os.path.exists("checkpoints/final.pt"):
        shutil.copy2("checkpoints/final.pt", "/checkpoints/final.pt")
        print("Saved final.pt to volume", flush=True)
        
        # Also return the checkpoint
        with open("checkpoints/final.pt", "rb") as f:
            return f.read()
    else:
        print("WARNING: No final.pt found!", flush=True)
        # Check what's in checkpoints
        if os.path.exists("checkpoints"):
            print(f"Checkpoints dir contents: {os.listdir('checkpoints')}", flush=True)
    
    return None


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/checkpoints": volume},
)
def generate(model_code: str, prompt: str = "", tokens: int = 500, temperature: float = 0.8):
    """Generate text from trained model."""
    import os
    import sys
    
    os.makedirs("/root/BloodMeridianNLP", exist_ok=True)
    os.chdir("/root/BloodMeridianNLP")
    
    # Write model code
    with open("model.py", "w") as f:
        f.write(model_code)
    
    if not os.path.exists("/checkpoints/final.pt"):
        return "No trained model found! Run training first."
    
    sys.path.insert(0, "/root/BloodMeridianNLP")
    import torch
    from model import McCarthyGPT
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load("/checkpoints/final.pt", map_location=device)
    config = ckpt['config']
    meta = ckpt['meta']
    
    model = McCarthyGPT(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    stoi = meta['stoi']
    itos = meta['itos']
    
    # Encode prompt
    if prompt:
        tokens_list = [stoi[c] for c in prompt if c in stoi]
    else:
        tokens_list = [0]
    
    x = torch.tensor([tokens_list], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=tokens, temperature=temperature)
    
    result = ''.join([itos[i] for i in generated[0].tolist()])
    return result


@app.local_entrypoint()
def main():
    import os
    
    base = "/Users/amadeuswoo/Documents/GitHub/BloodMeridianNLP"
    
    print("=" * 60)
    print("McCarthyGPT Training on Modal (T4 GPU)")
    print("=" * 60)
    
    # Load local files
    print("Loading local files...")
    
    with open(f"{base}/data/train.bin", "rb") as f:
        train_data = f.read()
    with open(f"{base}/data/val.bin", "rb") as f:
        val_data = f.read()
    with open(f"{base}/data/meta.pkl", "rb") as f:
        meta_data = f.read()
    with open(f"{base}/model.py", "r") as f:
        model_code = f.read()
    with open(f"{base}/train.py", "r") as f:
        train_code = f.read()
    
    print(f"  train.bin: {len(train_data):,} bytes")
    print(f"  val.bin: {len(val_data):,} bytes")
    print()
    
    # Run training
    checkpoint = train.remote(train_data, val_data, meta_data, model_code, train_code)
    
    if checkpoint:
        # Save checkpoint locally too
        os.makedirs(f"{base}/checkpoints", exist_ok=True)
        with open(f"{base}/checkpoints/final_modal.pt", "wb") as f:
            f.write(checkpoint)
        print(f"\nSaved checkpoint to {base}/checkpoints/final_modal.pt")
        
        print("\n" + "=" * 60)
        print("Generating sample...")
        print("=" * 60 + "\n")
        
        sample = generate.remote(model_code, prompt="The judge", tokens=500, temperature=0.8)
        print(sample)
    else:
        print("Training failed - no checkpoint produced")

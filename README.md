# 🏜️ BloodMeridianNLP (McCarthyGPT)

A character-level GPT trained on Cormac McCarthy's prose style. Generates text with that distinctive sparse, biblical, blood-soaked Western voice.

**Live demo**: [cmcgpt site](https://cmcgpt.vercel.app) · **Repo**: [BloodMeridianNLP](https://github.com/TheApexWu/BloodMeridianNLP)

## Quick Start

```bash
# Generate text
python play.py "The judge"

# Interactive mode
python play.py

# Web app
pip install gradio
python app.py
```

## Model Details

| Spec | Value |
|------|-------|
| Parameters | 4.81M |
| Architecture | GPT (6 layers, 6 heads, 384 dim) |
| Context | 256 characters |
| Vocab | 94 characters |
| Training | 5000 steps on T4 GPU |
| Final Loss | ~0.8 train / ~1.3 val |

## Files

```
BloodMeridianNLP/
├── model.py          # McCarthyGPT architecture
├── train.py          # Training loop
├── play.py           # Interactive CLI
├── app.py            # Gradio web interface
├── generate.py       # Simple generation script
├── evaluate.py       # Model evaluation
├── modal_train.py    # Modal cloud training wrapper
├── checkpoints/
│   └── final_modal.pt  # Trained model (57MB)
└── data/
    ├── train.bin     # Training data
    ├── val.bin       # Validation data
    └── meta.pkl      # Vocabulary
```

## Usage

### CLI Generation

```bash
# Single prompt
python play.py "They rode out at dawn"

# With options
python play.py --temp 0.5 --len 300 "The desert"

# Interactive REPL
python play.py
```

### Web App

```bash
pip install gradio
python app.py

# Public sharing link
python app.py --share
```

### Python API

```python
import torch
from model import McCarthyGPT

# Load
ckpt = torch.load('checkpoints/final_modal.pt', weights_only=False)
model = McCarthyGPT(ckpt['config'])
model.load_state_dict(ckpt['model'])
model.eval()

# Generate
stoi, itos = ckpt['meta']['stoi'], ckpt['meta']['itos']
prompt = "The judge"
x = torch.tensor([[stoi[c] for c in prompt]])
out = model.generate(x, max_new_tokens=500, temperature=0.8)
print(''.join([itos[i] for i in out[0].tolist()]))
```

## Training

Trained on Modal (cloud GPU):

```bash
pip install modal
modal setup
python -m modal run modal_train.py
```

Or locally:

```bash
python train.py
```

## Sample Output

**Prompt:** "The judge"

> The judge smiled. The grounds of his spirit around the company halted and Glanton filled his saddle and studied the alcalde's arms. Bueno, he said. They rode out across the pan and the imbecile started across the stony ground...

## Architecture

Based on Karpathy's nanoGPT, adapted for character-level generation:

- **Embedding:** 384-dim character embeddings + learned positional
- **Transformer:** 6 layers, 6 heads, causal attention
- **Output:** Softmax over 94-char vocabulary
- **Generation:** Autoregressive with temperature sampling

## License

Educational/research use. The training corpus style belongs to its original author.

## Credits

- Architecture inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)
- Training infrastructure: [Modal](https://modal.com)
- Built by [@theapexwu](https://github.com/theapexwu)

# BloodMeridianNLP (McCarthyGPT)

A character-level GPT trained on Cormac McCarthy's prose. Generates text with that sparse, biblical, blood-soaked Western voice.

**Live:** [cmcgpt.suzerain.dev](https://cmcgpt.suzerain.dev) | **Repo:** [GitHub](https://github.com/TheApexWu/BloodMeridianNLP)

---

## Quick Start

```bash
# Generate from CLI
python play.py "The judge"

# Interactive mode
python play.py

# Web app (Gradio)
pip install gradio
python app.py
```

---

## Model Versions

|  | v0 (Original) | v1 (Enhanced) |
|--|---------------|---------------|
| Parameters | 4.81M | ~4.8M |
| Layers | 6 | 8 |
| Heads | 6 | 8 |
| Embedding dim | 384 | 512 |
| Context window | 256 chars | 768 chars |
| Attention | Standard causal | ALiBi (linear bias) |
| Activation | GELU | SwiGLU |
| Training | 5000 steps, cosine LR | 8000 steps, curriculum learning |
| Dropout | 0.15 | 0.20 + stochastic depth |

v0 is the baseline trained on Modal (T4 GPU). v1 adds ALiBi attention for better long-range dependencies, SwiGLU activations, and a 3-stage curriculum that gradually increases sequence length (256 > 512 > 768).

---

## McCarthy's Fingerprint

Before training, we ran corpus analysis on Blood Meridian (633K characters, 117K words) to extract quantitative stylistic targets. These become the evaluation metrics for generated text.

| Metric | Corpus Value | What It Means |
|--------|-------------|---------------|
| Monosyllable % | 81.6% | 4 out of 5 words are one syllable. Drumbeat rhythm. |
| "And" frequency | 5.82% | Polysyndeton. "And...and...and" chaining as rhythmic device. |
| Avg sentence length | 15.7 words | Bimodal: short punches (1-10 words, 48%) mixed with long flows (31-50 words, 10%). |
| Quotation marks | ~0 | Dialogue marked by "said," never by quotes. |
| Avg syllables/word | 1.22 | Punchy monosyllabic vocabulary dominates. |
| Top sentence starters | "The" (18%), "He" (13%) | Character-centric, declarative. |
| "And" chains (3+) | 853 instances | The rhythmic engine of McCarthy's prose. |

### How the Model Scored (v0)

| Metric | Target | Generated | Verdict |
|--------|--------|-----------|---------|
| Monosyllable % | 81.6% | 78.5% | Close |
| Avg syllables/word | 1.22 | 1.26 | Close |
| "And" frequency | 5.82% | 12.08% | 2x high (overuses polysyndeton) |
| Quotation marks | ~0 | 0 | Perfect |
| Avg sentence length | 15.7 | 48.0 | 3x long (run-on sentences) |

**Diagnosis:** The model learned McCarthy's word-level patterns (monosyllables, no quotes, dark vocabulary) but not his sentence-level rhythm. With only 256-char context (~50 words), it can't see enough sentence structure to learn when to stop. v1's 768-char context (3x) directly addresses this.

Full corpus analysis with all 13 findings: [FINDINGS.md](FINDINGS.md)

---

## Project Structure

```
BloodMeridianNLP/
├── models/
│   ├── v0/
│   │   ├── model.py          # McCarthyGPT (4.81M, 6L/6H/384D)
│   │   └── train.py          # Training loop (cosine LR, early stopping)
│   └── v1/
│       ├── model.py          # RefinedMcCarthyGPT (ALiBi, SwiGLU, stochastic depth)
│       └── train.py          # Curriculum learning, gradient accumulation
│
├── play.py                   # Interactive CLI
├── app.py                    # Gradio web interface
├── generate.py               # Simple generation script
├── evaluate.py               # McCarthy-style metrics evaluation
├── modal_train.py            # Modal cloud training (T4 GPU)
│
├── prepare_data.py           # Corpus tokenization + train/val split
├── clean_corpus.py           # Corpus cleaning/normalization
├── corpus_analysis.py        # Statistical analysis of source text
│
├── site/
│   └── index.html            # cmcgpt.com static landing page
│
├── checkpoints/              # Trained model weights (.gitignored)
├── data/                     # Tokenized binary data (.gitignored)
├── corpus/                   # Source text
└── FINDINGS.md               # Full corpus analysis (13 findings)
```

---

## Usage

### CLI Generation

```bash
python play.py "They rode out at dawn"
python play.py --temp 0.5 --len 300 "The desert"

# Interactive REPL with /temp, /len, /quit commands
python play.py
```

### Web App

```bash
python app.py              # localhost:7860
python app.py --share      # public Gradio link
```

### Evaluation

```bash
# Generate samples and score against McCarthy metrics
python evaluate.py

# Evaluate your own text
python evaluate.py --text "They rode on through the dark..."
python evaluate.py --file some_output.txt
```

### Training

```bash
# Cloud (Modal, T4 GPU)
pip install modal && modal setup
python -m modal run modal_train.py

# Local
python models/v0/train.py    # original
python models/v1/train.py    # enhanced
```

---

## Sample Output

**Prompt:** "The judge"

> The judge smiled. The grounds of his spirit around the company halted and Glanton filled his saddle and studied the alcalde's arms. Bueno, he said. They rode out across the pan and the imbecile started across the stony ground...

---

## Architecture

Based on Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), adapted for character-level generation on a single novel.

v0: Standard GPT with learned positional embeddings, pre-norm transformer blocks, weight-tied output head. Autoregressive sampling with temperature and top-k.

v1: Replaces learned positions with ALiBi (no position embeddings needed, better extrapolation), swaps GELU for SwiGLU (gated activations), adds stochastic depth during training, and uses nucleus (top-p) sampling for more diverse generation.

---

## License

Educational/research use. The training corpus style belongs to its original author.

Built by [@theapexwu](https://github.com/theapexwu)

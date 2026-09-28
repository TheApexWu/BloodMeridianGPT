# BloodMeridianGPT (McCarthyGPT)

A character-level GPT trained on Cormac McCarthy's prose. Generates text with that sparse, biblical, blood-soaked Western voice.

**Live:** [cmcgpt.vercel.app](https://cmcgpt.vercel.app) | **Repo:** [GitHub](https://github.com/TheApexWu/BloodMeridianGPT)

---

## Quick Start

Run everything from the repository root.

```bash
# Generate from the CLI
python -m inference.play "The judge"

# Interactive mode
python -m inference.play

# Web app (Gradio)
pip install gradio
python -m inference.app
```

---

## Model Versions

|  | v0 (trained) | v1 (designed, not yet trained) |
|--|---------------|---------------|
| Parameters | 4.81M | — |
| Layers | 6 | 8 |
| Heads | 8 | 8 |
| Embedding dim | 256 | 512 |
| Context window | 256 chars | 768 chars |
| Attention | Standard causal | ALiBi (linear bias) |
| Activation | GELU | SwiGLU |
| Training | 5000 steps, cosine LR | 8000 steps, curriculum learning |
| Dropout | 0.1 | 0.20 + stochastic depth |

v0 is the baseline trained on Modal (T4 GPU); its weights are `checkpoints/final_modal.pt`. v1 adds ALiBi attention for better long-range dependencies, SwiGLU activations, and a 3-stage curriculum that gradually increases sequence length (256 > 512 > 768). No v1 checkpoint exists yet.

`checkpoints/best.pt` is an abandoned 384-dim retrain whose validation loss sits at chance (4.60 against ln 94 ≈ 4.54); nothing uses it. The v0 checkpoints pickled only `vocab_size` and `block_size`, so every loader rebuilds the rest with `restore_config()` in `models/v0/model.py`.

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

Full corpus analysis with all 13 findings: [docs/FINDINGS.md](docs/FINDINGS.md)

---

## Project Structure

```
BloodMeridianGPT/
├── models/
│   ├── v0/  model.py, train.py      # McCarthyGPT (4.81M, 6L/8H/256D) + restore_config()
│   └── v1/  model.py, train.py      # RefinedMcCarthyGPT (ALiBi, SwiGLU, stochastic depth)
│
├── inference/                        # run as python -m inference.<name>
│   ├── play.py                       # Interactive CLI
│   ├── app.py                        # Gradio web interface
│   ├── webapp.py                     # Alternative web interface
│   ├── generate.py                   # Simple generation script
│   └── export_onnx.py                # Export v0 to ONNX for the site
│
├── training/modal_train.py           # Modal cloud training (T4 GPU)
├── data_prep/                        # prepare_data, clean_corpus, repair_corpus, corpus_analysis
├── evaluation/                       # evaluate.py (McCarthy metrics), enhancement tests and plots
│
├── lib/                              # LLM style benchmark: API clients, metrics, prompts, statistics
├── notebooks/                        # 00 fingerprint · 01 contamination · 02 benchmark · 03 internals
├── judge/extract_judge.py            # Judge Holden speech extractor (see below)
│
├── site/                             # Static site with in-browser ONNX generation
├── docs/                             # FINDINGS, explanation guide, notes, figures
│
├── checkpoints/   (gitignored)       # Trained weights
├── data/          (gitignored)       # Tokenized data, benchmark passages, Judge outputs
└── corpus/        (gitignored)       # Source text
```

The novel is copyrighted, so the corpus, the benchmark's example passages (`data/benchmark/passages.json`) and every derived output stay out of git. Code that needs them reads them from those gitignored paths.

---

## Usage

### CLI Generation

```bash
python -m inference.play "They rode out at dawn"
python -m inference.play --temp 0.5 --len 300 "The desert"

# Interactive REPL with /temp, /len, /quit commands
python -m inference.play
```

### Web App

```bash
python -m inference.app              # localhost:7860
python -m inference.app --share      # public Gradio link
```

### Evaluation

```bash
# Generate samples and score against McCarthy metrics
python -m evaluation.evaluate

# Evaluate your own text
python -m evaluation.evaluate --text "They rode on through the dark..."
python -m evaluation.evaluate --file some_output.txt
```

### Training

```bash
# Cloud (Modal, T4 GPU)
pip install modal && modal setup
modal run training/modal_train.py

# Local
python models/v0/train.py    # original
python models/v1/train.py    # enhanced
```

---

## LLM Style Benchmark

`lib/` and `notebooks/` test whether frontier LLMs reproduce McCarthy's distributional fingerprint or only his surface markers: passage continuation, zero- and few-shot scene generation, and style transfer, scored with stylometric distances (Burrows' Delta, JSD on POS bigrams and sentence lengths), MAUVE, BERTScore and McCarthy-specific counts, with bootstrap confidence intervals. API keys load from the environment.

---

## The Judge (in progress)

A Judge Holden agent that invents his own parables instead of reciting the famous lines. `judge/extract_judge.py` recovers the Judge's speech from a novel with no quotation marks, tagging every segment with the rule that found it so precision can be measured per rule against hand labels. Famous scenes are held out as canaries: a model that produces them is reciting, not inventing.

```bash
python judge/extract_judge.py            # extract  -> data/judge/judge_segments.jsonl
python judge/extract_judge.py --gold     # blind labelling sheet
python judge/extract_judge.py --score    # precision and recall per rule
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

# Blood Meridian NLP

Train a language model on Cormac McCarthy's *Blood Meridian* to generate prose in his distinctive style.

## Project Status

- [x] Corpus extraction (clean text from PDF)
- [x] Spanish/English code-switching analysis (legacy)
- [ ] Data preparation (tokenization)
- [ ] nanoGPT training
- [ ] Generation & evaluation

## Corpus

`corpus/blood_meridian.txt` — Clean OCR extraction from Internet Archive (635KB, ~110K tokens)

**McCarthy's style features to capture:**
- No quotation marks for dialogue
- Long, unpunctuated sentences with polysyndeton ("and... and... and...")
- Biblical/archaic diction
- Spanish code-switching
- Paratactic structure (clauses joined without subordination)

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy tiktoken

# 2. Prepare data
python prepare_data.py

# 3. Train (coming soon)
python train.py
```

## Architecture

Using a modified nanoGPT approach:
- Character-level tokenizer (preserves McCarthy's punctuation style)
- Small transformer (~10-50M params)
- Trainable on consumer GPU or M-series Mac

## Legacy Code

The original project analyzed Spanish/English code-switching in the novel:
- `EnglishOrSpanish.py` — Naive Bayes classifier for language detection
- `Blood-Meridian-Spanish-Words-Model.txt` — Extracted Spanish vocabulary

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's minimal GPT implementation
- [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2-hour walkthrough
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original transformer paper

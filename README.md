# Blood Meridian NLP

Char-level language model trained on McCarthy's prose.

## Corpus

`corpus/blood_meridian.txt` — 635KB, ~110K tokens

See `FINDINGS.md` for stylistic analysis (syllables, polysyndeton, etc).

## Usage

```bash
pip install numpy pyphen

python prepare_data.py      # tokenize
python corpus_analysis.py   # analyze
python train.py             # train (wip)
```

## Architecture

- Character-level (preserves McCarthy's punctuation)
- nanoGPT-style transformer
- ~10-50M params, trainable on M-series Mac

## Key findings

- 81.6% monosyllables
- 5.82% "and" (polysyndeton)
- 0 quotation marks
- Bimodal sentence length (short punches + long flows)

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

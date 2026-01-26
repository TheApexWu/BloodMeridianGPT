#!/usr/bin/env python3
"""McCarthy corpus analysis. Run with --compare FILE to check generated text."""

import re
import argparse
from collections import Counter
from pathlib import Path

try:
    import pyphen
    DIC = pyphen.Pyphen(lang='en_US')
except ImportError:
    DIC = None


def syllables(word):
    if not DIC:
        return 1
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    h = DIC.inserted(word)
    return max(1, len(h.split('-'))) if h else 1


def load(path, skip=55):
    lines = open(path).readlines()
    return ''.join(lines[skip:])


def analyze(text, label=""):
    words = re.findall(r'[a-zA-Z]+', text)
    word_lower = [w.lower() for w in words]
    
    # syllables
    syls = [syllables(w) for w in words]
    mono_pct = sum(1 for s in syls if s == 1) / len(syls) * 100 if syls else 0
    avg_syl = sum(syls) / len(syls) if syls else 0
    
    # polysyndeton
    and_ct = sum(1 for w in word_lower if w == 'and')
    and_pct = and_ct / len(words) * 100 if words else 0
    
    # sentences
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    sent_lens = [len(s.split()) for s in sents]
    avg_sent = sum(sent_lens) / len(sent_lens) if sent_lens else 0
    
    # vocab
    unique = len(set(word_lower))
    
    # quotes (should be ~0 for McCarthy)
    quotes = text.count('"')
    
    print(f"\n{label or 'ANALYSIS'}")
    print(f"  words: {len(words):,} | unique: {unique:,}")
    print(f"  mono: {mono_pct:.1f}% | avg syl: {avg_syl:.2f}")
    print(f"  'and': {and_pct:.2f}% ({and_ct:,})")
    print(f"  sentences: {len(sents):,} | avg len: {avg_sent:.1f}")
    print(f"  quotes: {quotes}")
    
    return {
        'mono_pct': mono_pct,
        'and_pct': and_pct,
        'avg_sent': avg_sent,
        'quotes': quotes
    }


def compare(gen_path):
    gen = open(gen_path).read()
    
    print("McCARTHY BASELINE")
    print("  mono: 81.6% | 'and': 5.82% | quotes: ~0")
    
    m = analyze(gen, "GENERATED")
    
    print("\nDIFF")
    print(f"  mono: {m['mono_pct'] - 81.6:+.1f}%")
    print(f"  'and': {m['and_pct'] - 5.82:+.2f}%")
    print(f"  quotes: {m['quotes']} {'✓' if m['quotes'] == 0 else '✗'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compare', help='compare generated file to McCarthy')
    args = p.parse_args()
    
    if args.compare:
        compare(args.compare)
        return
    
    corpus = Path(__file__).parent / "corpus" / "blood_meridian.txt"
    if not corpus.exists():
        print(f"not found: {corpus}")
        return
    
    text = load(corpus)
    analyze(text, "BLOOD MERIDIAN")


if __name__ == '__main__':
    main()

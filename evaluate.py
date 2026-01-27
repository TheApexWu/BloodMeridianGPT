#!/usr/bin/env python3
"""Evaluate generated text against McCarthy's fingerprint."""

import re
import argparse
from collections import Counter

try:
    import pyphen
    DIC = pyphen.Pyphen(lang='en_US')
except ImportError:
    DIC = None

# McCarthy's baseline (from FINDINGS.md)
MCCARTHY = {
    'mono_pct': 81.6,
    'and_pct': 5.82,
    'avg_syl': 1.22,
    'quotes': 0,
}

def syllables(word):
    if not DIC:
        return 1
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    h = DIC.inserted(word)
    return max(1, len(h.split('-'))) if h else 1

def analyze(text):
    words = re.findall(r'[a-zA-Z]+', text)
    if not words:
        return None
    
    word_lower = [w.lower() for w in words]
    syls = [syllables(w) for w in words]
    
    mono_pct = sum(1 for s in syls if s == 1) / len(syls) * 100
    avg_syl = sum(syls) / len(syls)
    and_ct = sum(1 for w in word_lower if w == 'and')
    and_pct = and_ct / len(words) * 100
    quotes = text.count('"')
    
    return {
        'mono_pct': mono_pct,
        'and_pct': and_pct,
        'avg_syl': avg_syl,
        'quotes': quotes,
        'word_count': len(words),
    }

def score(metrics):
    """Score 0-100 based on how close to McCarthy's style."""
    if not metrics:
        return 0
    
    mono_diff = abs(metrics['mono_pct'] - MCCARTHY['mono_pct'])
    and_diff = abs(metrics['and_pct'] - MCCARTHY['and_pct'])
    syl_diff = abs(metrics['avg_syl'] - MCCARTHY['avg_syl'])
    quote_penalty = min(metrics['quotes'] * 5, 20)  # -5 per quote, max -20
    
    # Lower diff = better
    mono_score = max(0, 25 - mono_diff)
    and_score = max(0, 25 - and_diff * 5)
    syl_score = max(0, 25 - syl_diff * 50)
    quote_score = 25 - quote_penalty
    
    return mono_score + and_score + syl_score + quote_score

def evaluate(text, verbose=True):
    m = analyze(text)
    if not m:
        print("No text to analyze")
        return
    
    s = score(m)
    
    if verbose:
        print(f"\n{'='*50}")
        print("MCCARTHY SIMILARITY SCORE")
        print(f"{'='*50}")
        print(f"\nScore: {s:.1f}/100\n")
        print(f"{'Metric':<15} {'Generated':>10} {'McCarthy':>10} {'Diff':>10}")
        print("-" * 45)
        print(f"{'Monosyllables':<15} {m['mono_pct']:>9.1f}% {MCCARTHY['mono_pct']:>9.1f}% {m['mono_pct']-MCCARTHY['mono_pct']:>+9.1f}")
        print(f"{'And frequency':<15} {m['and_pct']:>9.2f}% {MCCARTHY['and_pct']:>9.2f}% {m['and_pct']-MCCARTHY['and_pct']:>+9.2f}")
        print(f"{'Avg syllables':<15} {m['avg_syl']:>10.2f} {MCCARTHY['avg_syl']:>10.2f} {m['avg_syl']-MCCARTHY['avg_syl']:>+10.2f}")
        print(f"{'Quotes':<15} {m['quotes']:>10} {MCCARTHY['quotes']:>10} {'✓' if m['quotes']==0 else '✗':>10}")
        print(f"\nWord count: {m['word_count']}")
    
    return s, m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('file', nargs='?', help='file to evaluate')
    p.add_argument('--text', '-t', help='text to evaluate directly')
    args = p.parse_args()
    
    if args.text:
        text = args.text
    elif args.file:
        text = open(args.file).read()
    else:
        print("Usage: evaluate.py <file> or --text 'text'")
        return
    
    evaluate(text)

if __name__ == '__main__':
    main()

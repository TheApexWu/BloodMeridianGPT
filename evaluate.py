#!/usr/bin/env python3
"""
Evaluate generated text against McCarthy's stylistic fingerprint.

Metrics from FINDINGS.md:
- Monosyllable percentage (target: ~81%)
- "And" frequency (target: ~5.8%)
- Sentence length distribution (bimodal)
- Quotation marks (target: 0)
- Average syllables per word (target: ~1.22)
"""

import re
import torch
import argparse
from collections import Counter

# =============================================================================
# SYLLABLE COUNTING
# =============================================================================

def count_syllables(word):
    """
    Estimate syllable count for a word.
    
    Simple heuristic: count vowel groups.
    Not perfect, but good enough for statistical analysis.
    """
    word = word.lower().strip()
    if not word:
        return 0
    
    # Handle common silent-e endings
    if word.endswith('e') and len(word) > 2:
        word = word[:-1]
    
    # Count vowel groups
    vowels = 'aeiouy'
    count = 0
    prev_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    
    return max(1, count)  # every word has at least 1 syllable


# =============================================================================
# TEXT ANALYSIS
# =============================================================================

def analyze_text(text):
    """
    Analyze text for McCarthy-style metrics.
    
    Returns dict with all metrics.
    """
    results = {}
    
    # Clean text
    text = text.strip()
    
    # --- Word-level analysis ---
    words = re.findall(r"[a-zA-Z']+", text)
    total_words = len(words)
    
    if total_words == 0:
        return {"error": "No words found in text"}
    
    results['total_words'] = total_words
    
    # Syllable analysis
    syllable_counts = [count_syllables(w) for w in words]
    monosyllables = sum(1 for s in syllable_counts if s == 1)
    
    results['monosyllable_pct'] = 100 * monosyllables / total_words
    results['avg_syllables'] = sum(syllable_counts) / total_words
    
    # Syllable distribution
    syl_dist = Counter(syllable_counts)
    results['syllable_dist'] = {k: 100 * v / total_words for k, v in sorted(syl_dist.items())}
    
    # --- "And" frequency ---
    and_count = sum(1 for w in words if w.lower() == 'and')
    results['and_frequency'] = 100 * and_count / total_words
    results['and_count'] = and_count
    
    # --- Punctuation analysis ---
    results['quotation_marks'] = text.count('"') + text.count('"') + text.count('"')
    results['periods'] = text.count('.')
    results['commas'] = text.count(',')
    
    # --- Sentence analysis ---
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if sentences:
        sentence_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
        sentence_lengths = [l for l in sentence_lengths if l > 0]
        
        if sentence_lengths:
            results['num_sentences'] = len(sentence_lengths)
            results['avg_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths)
            results['min_sentence_length'] = min(sentence_lengths)
            results['max_sentence_length'] = max(sentence_lengths)
            
            # Sentence length buckets (matching FINDINGS.md)
            buckets = {'1-10': 0, '11-20': 0, '21-30': 0, '31-50': 0, '51+': 0}
            for l in sentence_lengths:
                if l <= 10:
                    buckets['1-10'] += 1
                elif l <= 20:
                    buckets['11-20'] += 1
                elif l <= 30:
                    buckets['21-30'] += 1
                elif l <= 50:
                    buckets['31-50'] += 1
                else:
                    buckets['51+'] += 1
            
            results['sentence_length_dist'] = {
                k: 100 * v / len(sentence_lengths) for k, v in buckets.items()
            }
    
    # --- Vocabulary ---
    unique_words = len(set(w.lower() for w in words))
    results['unique_words'] = unique_words
    results['type_token_ratio'] = 100 * unique_words / total_words
    
    # --- Top words ---
    word_freq = Counter(w.lower() for w in words)
    results['top_words'] = word_freq.most_common(10)
    
    return results


def compare_to_mccarthy(results):
    """
    Compare results to McCarthy's baseline from FINDINGS.md.
    """
    targets = {
        'monosyllable_pct': 81.6,
        'avg_syllables': 1.22,
        'and_frequency': 5.82,
        'quotation_marks': 0,
        'avg_sentence_length': 15.7,
    }
    
    print("\n" + "=" * 60)
    print("COMPARISON TO McCARTHY'S STYLE")
    print("=" * 60)
    
    for metric, target in targets.items():
        if metric in results:
            actual = results[metric]
            diff = actual - target
            pct_diff = 100 * diff / target if target != 0 else float('inf')
            
            # Color coding (conceptual)
            if abs(pct_diff) < 10:
                status = "✓ CLOSE"
            elif abs(pct_diff) < 25:
                status = "~ OKAY"
            else:
                status = "✗ OFF"
            
            print(f"{metric:25s} | target: {target:6.2f} | actual: {actual:6.2f} | diff: {diff:+6.2f} | {status}")
    
    print("=" * 60)


def print_results(results):
    """Pretty print analysis results."""
    
    print("\n" + "=" * 60)
    print("TEXT ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📊 BASIC STATS")
    print(f"   Total words:      {results.get('total_words', 'N/A'):,}")
    print(f"   Unique words:     {results.get('unique_words', 'N/A'):,}")
    print(f"   Type-token ratio: {results.get('type_token_ratio', 0):.1f}%")
    print(f"   Sentences:        {results.get('num_sentences', 'N/A')}")
    
    print(f"\n📝 SYLLABLE ANALYSIS")
    print(f"   Monosyllables:    {results.get('monosyllable_pct', 0):.1f}%  (McCarthy: 81.6%)")
    print(f"   Avg syllables:    {results.get('avg_syllables', 0):.2f}   (McCarthy: 1.22)")
    
    if 'syllable_dist' in results:
        print(f"   Distribution:")
        for syl, pct in list(results['syllable_dist'].items())[:5]:
            bar = "█" * int(pct / 2)
            print(f"      {syl} syllable: {pct:5.1f}% {bar}")
    
    print(f"\n🔗 POLYSYNDETON ('AND')")
    print(f"   'And' frequency:  {results.get('and_frequency', 0):.2f}%  (McCarthy: 5.82%)")
    print(f"   'And' count:      {results.get('and_count', 0)}")
    
    print(f"\n✏️  PUNCTUATION")
    print(f"   Quotation marks:  {results.get('quotation_marks', 0)}      (McCarthy: ~0)")
    print(f"   Periods:          {results.get('periods', 0)}")
    print(f"   Commas:           {results.get('commas', 0)}")
    
    print(f"\n📏 SENTENCE LENGTH")
    print(f"   Average:          {results.get('avg_sentence_length', 0):.1f} words (McCarthy: 15.7)")
    print(f"   Range:            {results.get('min_sentence_length', 0)} - {results.get('max_sentence_length', 0)}")
    
    if 'sentence_length_dist' in results:
        print(f"   Distribution:")
        for bucket, pct in results['sentence_length_dist'].items():
            bar = "█" * int(pct / 2)
            print(f"      {bucket:6s} words: {pct:5.1f}% {bar}")
    
    print(f"\n🔤 TOP WORDS")
    if 'top_words' in results:
        for word, count in results['top_words']:
            print(f"      {word:12s} {count}")


# =============================================================================
# GENERATION + EVALUATION
# =============================================================================

def generate_and_evaluate(checkpoint_path, num_samples=5, tokens_per_sample=500, temperature=0.7):
    """
    Load model, generate samples, and evaluate against McCarthy metrics.
    """
    from model import McCarthyGPT
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    # Load meta from checkpoint or fallback to data/meta.pkl
    if 'meta' in ckpt:
        meta = ckpt['meta']
    else:
        import pickle
        print("Loading meta from data/meta.pkl...")
        with open('data/meta.pkl', 'rb') as f:
            meta = pickle.load(f)
    
    model = McCarthyGPT(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    if 'val_loss' in ckpt:
        print(f"Checkpoint val_loss: {ckpt['val_loss']:.4f}")
    if 'step' in ckpt:
        print(f"Checkpoint step: {ckpt['step']}")
    
    itos = meta['itos']
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples of {tokens_per_sample} tokens each...")
    print("-" * 60)
    
    all_text = []
    
    for i in range(num_samples):
        # Start with random seed from vocab
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated = model.generate(start, max_new_tokens=tokens_per_sample, temperature=temperature)
        
        text = ''.join([itos[idx] for idx in generated[0].tolist()])
        all_text.append(text)
        
        print(f"\n--- Sample {i+1} ---")
        print(text[:300] + "..." if len(text) > 300 else text)
    
    # Combine all samples for analysis
    combined = "\n\n".join(all_text)
    
    print("\n" + "=" * 60)
    print(f"ANALYZING {len(combined):,} characters of generated text")
    print("=" * 60)
    
    results = analyze_text(combined)
    print_results(results)
    compare_to_mccarthy(results)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate generated text against McCarthy metrics')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                        help='Analyze provided text instead of generating')
    parser.add_argument('--file', type=str, default=None,
                        help='Analyze text from file')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--tokens', type=int, default=500,
                        help='Tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    
    args = parser.parse_args()
    
    if args.text:
        # Analyze provided text
        results = analyze_text(args.text)
        print_results(results)
        compare_to_mccarthy(results)
    elif args.file:
        # Analyze text from file
        with open(args.file, 'r') as f:
            text = f.read()
        results = analyze_text(text)
        print_results(results)
        compare_to_mccarthy(results)
    else:
        # Generate and evaluate
        generate_and_evaluate(
            args.checkpoint,
            num_samples=args.samples,
            tokens_per_sample=args.tokens,
            temperature=args.temperature
        )

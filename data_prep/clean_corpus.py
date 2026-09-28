#!/usr/bin/env python3
"""
Clean the Blood Meridian corpus.

Fixes:
1. Remove PDF line wrapping (80-char breaks)
2. Preserve paragraph breaks (double newlines)
3. Normalize whitespace
4. Remove front matter (title pages, epigraphs)
"""

import re
import sys


def clean_corpus(input_path, output_path):
    with open(input_path, 'r') as f:
        text = f.read()
    
    # Stats before
    lines_before = text.count('\n')
    chars_before = len(text)
    
    # Find where the actual story starts (after epigraphs)
    # "See the child" is the first line of Chapter 1
    start_marker = "See the child"
    start_idx = text.find(start_marker)
    if start_idx > 0:
        text = text[start_idx:]
        print(f"Removed front matter ({start_idx} chars)")
    
    # Step 1: Normalize paragraph breaks
    # Double+ newlines = paragraph break (keep as \n\n)
    # Single newlines = PDF wrapping (convert to space)
    
    # First, standardize paragraph breaks to a placeholder
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # normalize multiple newlines
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    cleaned_paragraphs = []
    for para in paragraphs:
        # Within each paragraph, join wrapped lines
        para = para.replace('\n', ' ')
        # Normalize spaces
        para = re.sub(r'  +', ' ', para)
        para = para.strip()
        if para:
            cleaned_paragraphs.append(para)
    
    # Rejoin with double newlines
    text = '\n\n'.join(cleaned_paragraphs)
    
    # Step 2: Fix common OCR/extraction errors
    # (Add more as discovered)
    fixes = [
        (r'(\w)- (\w)', r'\1\2'),  # broken hyphens: "some- thing" -> "something"
        (r' +', ' '),              # multiple spaces
        (r' ([.,;:!?])', r'\1'),   # space before punctuation
    ]
    
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)
    
    # Stats after
    lines_after = text.count('\n')
    chars_after = len(text)
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(text)
    
    print(f"\nCleaning complete:")
    print(f"  Lines: {lines_before:,} -> {lines_after:,}")
    print(f"  Chars: {chars_before:,} -> {chars_after:,}")
    print(f"  Paragraphs: {len(cleaned_paragraphs):,}")
    print(f"\nSaved to: {output_path}")
    
    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE (first 1000 chars):")
    print("=" * 60)
    print(text[:1000])
    
    return text


if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'corpus/blood_meridian.txt'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'corpus/blood_meridian_clean.txt'
    clean_corpus(input_path, output_path)

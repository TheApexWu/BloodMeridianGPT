#!/usr/bin/env python3
"""
Repair OCR artifacts in Blood Meridian corpus.
Run: python repair_corpus.py
"""

import re
from pathlib import Path

# Every known OCR corruption mapped to its correction.
# Format: (corrupted, correct)
# Verified against published text context where possible.
FIXES = [
    # Line 51 - judge accusing Reverend Green
    ("years\u00ae! said eleven#=ho", "years\u2014said eleven\u2014who"),
    # Line 63
    ("cloth#:ng", "clothing"),
    # Line 1793
    ("guar#;nteed", "guaranteed"),
    # Line 1881 - judge's speech about existents
    ("extant2!s", "existents"),
    ("them\u2122ike", "them like"),
    ("chilgren", "children"),
    # Line 1901 - judge on facts and destiny
    ("with=sut", "without"),
    ("facts3so0 the extent that they can be readily made to do so\" hould", "facts\u2014to the extent that they can be readily made to do so\u2014should"),
    # Line 1901 - "authority transcends" broken across paragraph boundary
    ("authority trans\n\nends his", "authority transcends his"),
    # Line 1905
    ("for3\u00aeally", "formally"),
    # Line 1915
    ("fugizxive/n", "fugitive an"),
    ("he\u00aeJnd", "he and"),
    # Line 2065
    ("un# een", "unseen"),
    # Line 2091
    ("ex\u20acanse", "expanse"),
    # Line 2309
    ("bar2\u00aean", "barman"),
    # Line 2439
    ("watch#\u00a2ng", "watching"),
    # Line 2593
    ("dis\u20acosition", "disposition"),
    # Line 2875
    ("here#;bouts", "hereabouts"),
    # Line 2921
    ("strugling", "struggling"),
    ("in\u00a5ferits", "inherits"),
    ("wanzering", "wandering"),
    # Line 2967
    ("Glandzon", "Glanton"),
    ("moun'xains", "mountains"),
    # Line 3207
    ("inZjugurated", "inaugurated"),
    ("invenjxory", "inventory"),
    ("surpris#<ngly", "surprisingly"),
    ("considerble", "considerable"),
    # Line 3409
    ("reoved", "removed"),
    # Line 3411
    ("lurch#:ng", "lurching"),
    # Line 3413 - stray underscore
    ("and _ they", "and they"),
    # Line 3573
    ("animals?known", "animals\u2014known"),
    ("some#;oward", "some toward"),
    # Line 3751
    ("horse##an on the grounds", "horseman on the grounds"),
    # Line 3835
    ("mounzgains", "mountains"),
    ("horse##en rode", "horsemen rode"),
    # Line 3907
    ("selec3zion", "selection"),
    # Line 3909
    ("animals?goats", "animals\u2014goats"),
    ("burro\u00ae;hat", "burro\u2014that"),
    # Line 4317
    ("dis##antled", "dismantled"),
    # Line 4319
    ("attenjzion", "attention"),
    # Line 4385
    ("show%\u00a2ng", "showing"),
    ("democacy", "democracy"),
    # Line 4387
    ("fugijzives", "fugitives"),
    # Line 4495
    ("Glanton''s", "Glanton's"),
    ("hootng", "hooting"),
    # Line 4589
    ("amphixxheatre", "amphitheatre"),
    ("himself#-uch", "himself\u2014such"),
    ("com2oner", "commoner"),
    ("night2Jnd", "night\u2014and"),
    ("drowntng", "drowning"),
    ("shufled", "shuffled"),
    # Line 4813
    ("husgand", "husband"),
    ("pray#<ng", "praying"),
    # Line 4843 - word broken across paragraph boundary
    ("his re\u00ae\n\nentless jaw", "his relentless jaw"),
    # Line 4843
    ("pargels", "parcels"),
    ("cluszered", "clustered"),
    # Line 4931
    ("himelf", "himself"),
    # Line 4933
    ("peri#;dic", "periodic"),
    # Line 4961
    ("meas\u00a2ring", "measuring"),
    # Line 5023
    ("watch\u00a5ng", "watching"),
    # Line 5077
    ("whis\u20acering", "whispering"),
    # Line 5089
    ("thouz\u00a2and", "thousand"),
    ("sadtled", "saddled"),
    ("comanions", "companions"),
    # Line 5127
    ("ani##al", "animal"),
    # Line 5161
    ("Uptream", "Upstream"),
    ("drink#&ng", "drinking"),
    # Line 5167
    ("hold#& ng", "holding"),
    ("predaszors", "predators"),
    ("any=here", "anywhere"),
    # Line 5169
    ("ribtines", "rib-tines"),
    # Line 2913 - comenced (missing 'm')
    ("comenced", "commenced"),
    # Line 3625
    ("Glanszon", "Glanton"),
    ("preprations", "preparations"),
    # Line 3627
    ("Dela\u00a5are", "Delaware"),
]

# Chapter numbering fixes
# Blood Meridian has 23 chapters + epilogue
# Missing: X, XIII (listed as XI again), XVIII
CHAPTER_FIXES = [
    # These need to be verified against actual chapter positions
    # The second "XI" should be "XIII"
]


def repair(text):
    count = 0
    for corrupt, correct in FIXES:
        if corrupt in text:
            occurrences = text.count(corrupt)
            text = text.replace(corrupt, correct)
            count += occurrences
            print(f"  FIXED ({occurrences}x): {repr(corrupt)[:60]} -> {repr(correct)[:60]}")
        else:
            # Try to find near-misses (encoding differences)
            pass

    return text, count


def scan_remaining(text):
    """Find any remaining suspicious characters after fixes."""
    issues = []

    # Non-ASCII characters (except common ones like em-dash, smart quotes)
    allowed = set('\u2014\u2013\u2018\u2019\u201c\u201d\u00e9\u00f1\u00ed\u00e1\u00fa\u00fc\u00f3')
    for i, char in enumerate(text):
        if ord(char) > 127 and char not in allowed and char != '\n':
            start = max(0, i - 30)
            end = min(len(text), i + 30)
            context = text[start:end].replace('\n', ' ')
            line_num = text[:i].count('\n') + 1
            issues.append(f"  Line ~{line_num}: {repr(char)} (U+{ord(char):04X}) in: ...{context}...")

    # Hash adjacent to letters (OCR artifact pattern)
    for m in re.finditer(r'[a-zA-Z]#[^\s#]', text):
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 20)
        context = text[start:end].replace('\n', ' ')
        line_num = text[:m.start()].count('\n') + 1
        issues.append(f"  Line ~{line_num}: hash-in-word: ...{context}...")

    return issues


def main():
    corpus_dir = Path(__file__).resolve().parent.parent / "corpus"
    input_path = corpus_dir / "blood_meridian_clean.txt"
    output_path = corpus_dir / "blood_meridian_v2.txt"

    if not input_path.exists():
        print(f"Not found: {input_path}")
        return

    text = input_path.read_text()
    print(f"Input: {len(text):,} chars, {text.count(chr(10)):,} lines")
    print(f"\nApplying {len(FIXES)} fixes...")

    text, fix_count = repair(text)
    print(f"\nTotal replacements: {fix_count}")

    # Normalize em-dashes (some might be double hyphens, some unicode)
    # McCarthy uses minimal punctuation but does use dashes occasionally

    print(f"\nScanning for remaining issues...")
    issues = scan_remaining(text)
    if issues:
        print(f"Found {len(issues)} remaining suspicious characters:")
        for issue in issues[:30]:
            print(issue)
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more")
    else:
        print("  Clean!")

    # Stats
    words = re.findall(r'[a-zA-Z]+', text)
    print(f"\nOutput: {len(text):,} chars, {len(words):,} words")

    output_path.write_text(text)
    print(f"\nSaved to: {output_path}")
    print("Review the output, then replace blood_meridian_clean.txt if satisfied.")


if __name__ == '__main__':
    main()

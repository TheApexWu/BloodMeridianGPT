# Blood Meridian: Corpus Analysis Findings

> Exploratory Data Analysis before model training.  
> Generated: 2026-01-26

---

## Corpus Overview

| Metric | Value |
|--------|-------|
| Total characters | 633,463 |
| Total words | 117,484 |
| Unique words | 10,323 |
| Total sentences | 7,366 |
| Total paragraphs | 2,600 |

---

## Finding 1: Punctuation Patterns

McCarthy's signature: **no quotation marks for dialogue.**

| Punctuation | Count | Per 1000 chars |
|-------------|-------|----------------|
| Period `.` | 7,205 | 11.4 |
| Comma `,` | 1,939 | 3.1 |
| Apostrophe `'` | 969 | 1.5 |
| Question `?` | 609 | 1.0 |
| Hyphen `-` | 460 | 0.7 |
| Colon `:` | 17 | 0.03 |
| Semicolon `;` | 9 | 0.01 |
| **Double quotes `"`** | **2** | **~0** |

**Implication:** Character-level tokenization will naturally learn this. No special handling needed.

---

## Finding 2: Syllabic Structure

**McCarthy's secret: 81.6% monosyllables.**

| Syllables | Count | Percentage |
|-----------|-------|------------|
| 1 | 96,112 | 81.6% |
| 2 | 17,706 | 15.0% |
| 3 | 3,330 | 2.8% |
| 4 | 495 | 0.4% |
| 5+ | 88 | 0.1% |

**Average syllables per word: 1.22**

Monosyllable runs (consecutive 1-syllable words) are common:
- 5 in a row: 1,382 occurrences
- 6 in a row: 1,131 occurrences  
- 7 in a row: 910 occurrences
- 8 in a row: 687 occurrences

**Implication:** Generated text should maintain ~80% monosyllables and frequent mono runs. Use as evaluation metric.

---

## Finding 3: Sentence Length Distribution

Bimodal distribution — short punchy sentences + long flowing ones.

| Word count | Sentences | Percentage |
|------------|-----------|------------|
| 1-10 | 3,546 | 48.1% |
| 11-20 | 1,873 | 25.4% |
| 21-30 | 1,020 | 13.8% |
| 31-50 | 719 | 9.8% |
| 51-75 | 160 | 2.2% |
| 76-100 | 27 | 0.4% |
| 101-200 | 19 | 0.3% |
| 201+ | 2 | ~0% |

**Average: 15.7 words/sentence | Median: 11 | Max: 245**

**Implication:** Model should produce mix of short and long sentences, not uniform length.

---

## Finding 4: Polysyndeton ("And" Usage)

McCarthy's rhythmic device: chaining with "and."

- Total "and" occurrences: **6,787**
- Frequency: **5.82% of all words**
- Average per sentence: **0.92**
- "And...and...and" chains (3+): **853 instances**

**Implication:** High "and" frequency is a key stylistic marker. Track in evaluation.

---

## Finding 5: Vocabulary Richness

| Metric | Value |
|--------|-------|
| Total words | 117,484 |
| Unique words | 10,323 |
| Type-token ratio | 8.79% |
| Hapax legomena | 5,416 (52.5% of vocabulary) |

**Top 10 words:** the (11,115), and (6,791), of (3,110), a (2,679), he (2,673), in (2,392), to (1,983), they (1,681), his (1,625), with (1,210)

**Implication:** Over half the vocabulary appears only once — model needs to handle rare words.

---

## Finding 6: Paragraph Structure

| Metric | Value |
|--------|-------|
| Total paragraphs | 2,600 |
| Average words/paragraph | 44.4 |
| Median length | 20 words |
| Range | 3 - 481 words |

**Implication:** High variance in paragraph length. Model should learn paragraph breaks as rhythm markers.

---

## Finding 7: Sentence Starters

Most common first words:

| Starter | Count | % |
|---------|-------|---|
| The | 1,407 | 18.2% |
| He | 1,007 | 13.0% |
| They | 584 | 7.5% |
| I | 268 | 3.5% |
| A | 193 | 2.5% |
| Glanton | 157 | 2.0% |
| When | 154 | 2.0% |

**Implication:** "The" and "He" dominate. Model should mirror this distribution.

---

## Finding 8: Dialogue Without Quotes

McCarthy uses `said` as primary dialogue marker:

| Verb | Count |
|------|-------|
| said | 608 |
| called | 113 |
| told | 42 |
| cried | 20 |
| asked | 19 |
| whispered | 13 |

Dialogue blends into narrative without punctuation markers.

**Implication:** Model must learn contextual dialogue recognition, not rely on quotes.

---

## Finding 9: Archaic Vocabulary

McCarthy uses biblical/archaic diction sparingly but distinctively:

| Word | Count |
|------|-------|
| ye | 96 |
| wont | 22 |
| yonder | 19 |
| aught | 4 |
| yon | 3 |
| firmament | 3 |
| nigh | 2 |
| thee/thy | 4 |

**Total archaic instances: 160**

**Implication:** Archaic words are rare but important. Model should use them sparingly.

---

## Finding 10: Spanish Code-Switching

Spanish words are sparse (as expected):

| Word | Count |
|------|-------|
| nada | 7 |
| donde | 7 |
| quien | 7 |
| bueno | 6 |
| todo | 6 |
| como | 6 |

**Total Spanish instances: ~59**

**Implication:** Spanish is decorative, not structural. No special handling needed.

---

## Finding 11: Distinctive Phrases

| Bigram | Count |
|--------|-------|
| "the judge" | 458 |
| "the kid" | 368 |
| "he said" | 281 |
| "said the" | 197 |
| "they rode" | 151 |
| "among the" | 138 |
| "he looked" | 130 |
| "across the" | 116 |

**Implication:** Character-centric bigrams dominate. Model will learn these naturally.

---

## Finding 12: Motion Patterns

McCarthy's prose is driven by movement:

| Phrase | Count |
|--------|-------|
| "they rode" | 149 |
| "rode on" | 56 |
| "went on" | 46 |
| "rode out" | 42 |
| "they passed" | 38 |
| "they crossed" | 28 |
| "rode through" | 25 |

**Implication:** "Rode" and motion verbs are core to the rhythm.

---

## Finding 13: Color Vocabulary

McCarthy's palette is dark:

| Color | Count |
|-------|-------|
| dark | 198 |
| black | 117 |
| blood | 96 |
| brown | 89 |
| white | 80 |
| pale | 66 |
| blue | 50 |
| gray | 46 |
| red | 33 |
| silver | 23 |

**Implication:** Dark/black/blood dominate. Bright colors (yellow: 11, green: 19) are rare.

---

## Summary: McCarthy's Fingerprint

1. **No quotation marks** — dialogue marked by "said"
2. **81% monosyllables** — punchy, drumbeat rhythm
3. **5.8% "and"** — polysyndeton as rhythmic device
4. **Bimodal sentences** — short punches + long flows
5. **Dark palette** — black, blood, dark, pale
6. **Motion-driven** — "they rode", "rode on"
7. **Sparse archaisms** — ye, yonder, wont (but important)
8. **Minimal Spanish** — decorative, not structural

---

## Evaluation Metrics (for generated text)

- [ ] Monosyllable percentage (target: ~80%)
- [ ] "And" frequency (target: ~5.8%)
- [ ] Sentence length distribution (bimodal)
- [ ] Quotation mark count (target: 0)
- [ ] Archaic word presence (sparse but present)
- [ ] Color vocabulary distribution
- [ ] Motion phrase frequency

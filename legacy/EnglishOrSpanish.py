#!/usr/bin/env python3
"""Spanish word detection using char n-gram classifier."""

import os
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

BASE = Path(__file__).parent.parent

def load_words(path):
    return [line.strip().lower() for line in open(path)]

spanish = load_words(BASE / "spanish_dictionary.txt")
english = load_words(BASE / "english_dictionary.txt")

words = spanish + english
labels = ["spanish"] * len(spanish) + ["english"] * len(english)

print(f"loaded {len(spanish)} spanish, {len(english)} english")

X_train, X_test, y_train, y_test = train_test_split(
    words, labels, test_size=0.2, random_state=42
)

model = make_pipeline(
    CountVectorizer(analyzer="char", ngram_range=(2, 4)),
    MultinomialNB()
)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

# apply to corpus
text = open(BASE / "corpus" / "blood_meridian.txt").read()
corpus_words = re.findall(r"\b[a-zA-Z]+\b", text)
preds = model.predict(corpus_words)

spanish_found = [w for w, p in zip(corpus_words, preds) if p == "spanish"]
print(f"found {len(spanish_found)} spanish words in corpus")

# save
out = BASE / "Blood-Meridian-Spanish-Words-Model.txt"
open(out, "w").write("\n".join(spanish_found))

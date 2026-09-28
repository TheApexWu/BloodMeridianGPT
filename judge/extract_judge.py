#!/usr/bin/env python3
"""Extract the Judge's speech from Blood Meridian as (scene, discourse, location) records.

McCarthy uses no quotation marks, so speech is recovered from attribution tags and from
who the narration last named. Every record carries the rule that found it, so precision
is measured per rule against hand labels instead of trusted in aggregate.

    python3 judge/extract_judge.py            # extract     -> data/judge/judge_segments.jsonl
    python3 judge/extract_judge.py --gold     # blind sheet -> data/judge/gold_sheet.csv
    python3 judge/extract_judge.py --score    # score the labeled sheet

Everything written lands in data/judge/, which .gitignore keeps out of git: it is the novel's text.
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "blood_meridian_clean.txt"
OUT = ROOT / "data" / "judge"

JUDGE_TAG = re.compile(r",?\s*\b(?:said (?:the judge|holden)|(?:the judge|holden) said)\b\s*,?", re.I)
HE_TAG = re.compile(r",?\s*\b(?:he said|said he)\b\s*,?", re.I)
# "said <someone>" where someone is not the judge and not a pronoun or adverb. The corpus has
# OCR damage ("said living" for "said Irving"), so lowercase names must count as speakers too.
NOT_A_SPEAKER = {"the judge", "holden", "he", "she", "they", "it", "nothing", "that", "so", "to",
                 "in", "again", "softly", "quietly", "no", "yes", "aloud", "something", "not", "was"}
OTHER_TAG = re.compile(r"\bsaid ((?:the )?[A-Za-z]+)\b|\b((?:the )?[A-Z][a-z]+) said\b")
SUBJECT_JUDGE = re.compile(r"^(?:the judge|holden)\b", re.I)
SUBJECT_OTHER = re.compile(r"^(?:[A-Z][a-z]+(?: [A-Z][a-z]+)?|the (?!judge)[a-z]+)\b")
PAST = (r"\w+ed|said|sat|stood|rode|turned|spat|wrote|smiled|took|made|held|put|went|came|saw|lay|"
        r"rose|drew|gave|told|knew|found|brought|began|ran|set|left|got|kept|looked|nodded|shook|"
        r"studied|folded|pressed|placed|passed|laid|broke|tapped|fell|stopped|reached|spoke|heard|"
        r"thought|felt|seemed|led|sent|struck|hung|wore|threw|grew|drank|slept|woke|caught|shot|lurched")
NARRATION = re.compile(rf"^(?:the judge|holden|he|she|they|it|[A-Z][a-z]+(?: [A-Z][a-z]+)?|the [a-z]+)"
                       rf"\s+(?:\w+\s+){{0,2}}?(?:{PAST})\b", re.I)
PAST_ANY = re.compile(rf"\b(?:{PAST})\b", re.I)
SPEECH_SIGNAL = re.compile(r"\b(?:i|me|my|mine|we|us|our|you|your|ye|is|are|has|have|does|do|will|"
                           r"shall|can|cannot|must|may)\b|\?$", re.I)
ROMAN = re.compile(r"^[IVXL]+\.?$")
GENERAL = re.compile(r"\b(?:all|every|no man|men|man|war|whatever|nothing|world|god|law|game|"
                     r"order|existence|creation|truth|nature)\b", re.I)
# Scenes holding the lines everyone already knows. Held out: never train on them; they are
# the canary that tells you a model is reciting rather than inventing.
CANARY = ["whatever in creation", "without my consent", "men are born for games", "war is god",
          "moral law is an invention", "suzerain", "the truth about the world", "books lie",
          "anything is possible"]


def paragraphs(text):
    """Blank-line paragraphs, with PDF page breaks repaired: a paragraph that ends mid-sentence
    or a next one that starts lowercase is the same paragraph broken across a page."""
    raw = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in raw:
        if out and not ROMAN.match(p) and not ROMAN.match(out[-1]) and (
                not re.search(r"[.?!]['\"]?$", out[-1]) or p[:1].islower()):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def sentences(p):
    return [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z¿¡])", p) if s]


def other_tag(s):
    for m in OTHER_TAG.finditer(s):
        who = (m.group(1) or m.group(2) or "").lower()
        if who and who not in NOT_A_SPEAKER:
            return who
    return None


def is_narration(s):
    """Third person past with nothing speech-like in it. The subject need not come first:
    'In the night a caravan passed' is narration too."""
    bare = JUDGE_TAG.sub(" ", s)
    return not SPEECH_SIGNAL.search(bare) and bool(NARRATION.match(s) or PAST_ANY.search(bare))


def strip_tag(s):
    s = HE_TAG.sub(" ", JUDGE_TAG.sub(" ", s))
    return re.sub(r"\s+([,.?!])", r"\1", re.sub(r"\s{2,}", " ", s)).strip(" ,")


def extract(P):
    """One pass of speaker tracking. `named` is who the narration last put forward, which is
    what a bare 'he said' and an untagged reply resolve against."""
    recs, chapter, named, prev_judge_speech, prev_other_speech, judge_near = [], "I", None, False, False, -99
    for i, p in enumerate(P):
        if ROMAN.match(p):
            chapter, named, prev_judge_speech, prev_other_speech = p.rstrip("."), None, False, False
            continue
        sents, hits, speaker, narr_in_para, tagged_other = sentences(p), [], None, False, False
        for s in sents:
            if JUDGE_TAG.search(s):
                hits.append((strip_tag(s), "R1_tag")); speaker = "judge"
            elif other_tag(s):
                speaker, tagged_other = "other", True
            elif HE_TAG.search(s):
                speaker = "judge" if named == "judge" else "other"
                if speaker == "judge":
                    hits.append((strip_tag(s), "R2_pronoun"))
            elif is_narration(s):
                narr_in_para = True
                named = ("judge" if SUBJECT_JUDGE.search(s) else
                         "other" if SUBJECT_OTHER.search(s) and not s.lower().startswith(("he ", "they ")) else named)
                speaker = "judge" if named == "judge" else None
                if named == "judge":
                    judge_near = i
            elif speaker == "judge":
                hits.append((strip_tag(s), "R3_lead" if narr_in_para else "R1_cont"))
            elif speaker is None and not narr_in_para and prev_judge_speech and not tagged_other:
                if len(p.split()) >= 15:
                    hits.append((s, "R4_monologue"))
                else:
                    # a short untagged line right after the Judge is the other side of the exchange
                    speaker, tagged_other = "other", True
            elif (speaker is None and not narr_in_para and prev_other_speech
                  and i - judge_near <= 3 and not tagged_other):
                hits.append((s, "R5_reply"))
        if hits:
            text = " ".join(h[0] for h in hits).strip()
            rules = sorted({h[1] for h in hits})
            recs.append({"para": i, "chapter": chapter, "rules": rules,
                         "confidence": "low" if rules == ["R5_reply"] else "high",
                         "kind": "discourse" if len(text.split()) >= 18 or GENERAL.search(text) else "utterance",
                         "held_out": any(c in text.lower() for c in CANARY), "text": text})
            judge_near = i
        prev_judge_speech = bool(hits) and not tagged_other
        prev_other_speech = tagged_other and not hits
    for r in recs:  # the scene: up to ~900 chars of what came before, the prompt a model answers
        before, j = [], r["para"] - 1
        while j >= 0 and sum(map(len, before)) < 900 and not ROMAN.match(P[j]):
            before.insert(0, P[j]); j -= 1
        r["scene"] = " ".join(before)[-900:]
    return recs


def write_gold(P, recs, n=50, seed=1873):
    """Half extracted paragraphs (precision), half near-miss paragraphs within 4 of a judge
    mention that were NOT extracted (recall). The extractor's verdict goes in a separate key
    file so the labeler never sees it."""
    rng = random.Random(seed)
    got = {r["para"] for r in recs}
    near = sorted({j for i, p in enumerate(P) if re.search(r"\bjudge\b", p, re.I)
                   for j in range(max(0, i - 4), min(len(P), i + 5))
                   if j not in got and not ROMAN.match(P[j]) and len(P[j]) > 20})
    pick = rng.sample(sorted(got), n // 2) + rng.sample(near, n - n // 2)
    rng.shuffle(pick)
    with open(OUT / "gold_sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "para", "before", "paragraph", "judge_speaks (y/n)", "notes"])
        for k, i in enumerate(pick, 1):
            w.writerow([k, i, P[i - 1][-240:] if i else "", P[i], "", ""])
    json.dump({str(i): (i in got) for i in pick}, open(OUT / "gold_key.json", "w"))


def score():
    key = json.load(open(OUT / "gold_key.json"))
    rows = list(csv.DictReader(open(OUT / "gold_sheet.csv")))
    labeled = [r for r in rows if r["judge_speaks (y/n)"].strip().lower()[:1] in ("y", "n")]
    if len(labeled) < len(rows):
        print(f"{len(rows) - len(labeled)} of {len(rows)} rows unlabeled; scoring the {len(labeled)} labeled")
    tp = sum(1 for r in labeled if key[r["para"]] and r["judge_speaks (y/n)"].lower().startswith("y"))
    fp = sum(1 for r in labeled if key[r["para"]] and r["judge_speaks (y/n)"].lower().startswith("n"))
    fn = sum(1 for r in labeled if not key[r["para"]] and r["judge_speaks (y/n)"].lower().startswith("y"))
    tn = len(labeled) - tp - fp - fn
    print(f"tp {tp}  fp {fp}  fn {fn}  tn {tn}")
    print(f"precision {tp / max(1, tp + fp):.2f}  (of what it grabbed, share that is the Judge)")
    print(f"recall    {tp / max(1, tp + fn):.2f}  (of the Judge's near-miss lines, share it caught)")
    recs = {str(json.loads(l)["para"]): json.loads(l) for l in open(OUT / "judge_segments.jsonl")}
    by_rule = {}
    for r in labeled:
        for rule in recs.get(r["para"], {}).get("rules", []):
            y = r["judge_speaks (y/n)"].lower().startswith("y")
            by_rule.setdefault(rule, [0, 0])[0 if y else 1] += 1
    for rule, (y, n) in sorted(by_rule.items()):
        print(f"  {rule:<14} right {y}  wrong {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.score:
        return score()
    P = paragraphs(CORPUS.read_text(encoding="utf-8"))
    recs = extract(P)
    with open(OUT / "judge_segments.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.gold:
        write_gold(P, recs)
    rules = {}
    for r in recs:
        for x in r["rules"]:
            rules[x] = rules.get(x, 0) + 1
    words = sum(len(r["text"].split()) for r in recs)
    print(f"{len(P)} paragraphs -> {len(recs)} judge segments, {words:,} words")
    print("  by rule:", dict(sorted(rules.items())))
    print("  kind:", {k: sum(r["kind"] == k for r in recs) for k in ("discourse", "utterance")},
          " held out:", sum(r["held_out"] for r in recs))
    print("  canaries found:", sorted({c for r in recs for c in CANARY if c in r["text"].lower()}))


if __name__ == "__main__":
    main()

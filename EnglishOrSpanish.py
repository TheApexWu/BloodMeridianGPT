import os
import re 

# Define base path
base_path = r"C:\Users\AlexWu\Documents\GitHub\BloodMeridianNLP"

# Load word lists
def load_words(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines()]

# Load Spanish words
spanish_words = load_words(os.path.join(base_path, "spanish_dictionary.txt"))

# Load English words
english_words = load_words(os.path.join(base_path, "english_dictionary.txt"))

# Combine and label the data
words = spanish_words + english_words
labels = ["spanish"] * len(spanish_words) + ["english"] * len(english_words)

print(f"✅ Loaded {len(spanish_words)} Spanish words and {len(english_words)} English words.")


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(words, labels, test_size=0.2, random_state=42)

# Create a character-level n-gram model
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 4))  # Use 2-4 character n-grams
model = make_pipeline(vectorizer, MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Load *Blood Meridian* text
input_text_path = os.path.join(base_path, "Blood-Meridian.txt")
with open(input_text_path, "r", encoding="utf-8") as f:
    text = f.read()

# Extract all words
words_in_text = re.findall(r"\b[a-zA-Z]+\b", text)

# Predict language for each word
predictions = model.predict(words_in_text)

# Extract Spanish words
spanish_words_in_text = [word for word, pred in zip(words_in_text, predictions) if pred == "spanish"]

# Save Spanish words
output_path = os.path.join(base_path, "Blood-Meridian-Spanish-Words-Model.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(spanish_words_in_text))

print(f"✅ Extracted {len(spanish_words_in_text)} Spanish words using the model.")
print(f"✅ Results saved to: {output_path}")

# Split text into lines
lines = text.splitlines()

# Identify lines with a high density of Spanish words
spanish_dialogue = []
for line in lines:
    words_in_line = re.findall(r"\b[a-zA-Z]+\b", line)
    if not words_in_line:
        continue

    # Predict language for each word in the line
    predictions = model.predict(words_in_line)

    # Calculate the percentage of Spanish words
    spanish_percentage = sum(1 for pred in predictions if pred == "spanish") / len(words_in_line)

    # If more than 50% of the words are Spanish, assume it's dialogue
    if spanish_percentage > 0.5:
        spanish_dialogue.append(line)

# Save Spanish dialogue
dialogue_output_path = os.path.join(base_path, "Blood-Meridian-Spanish-Dialogue-Model.txt")
with open(dialogue_output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(spanish_dialogue))

print(f"✅ Extracted {len(spanish_dialogue)} lines of Spanish dialogue using the model.")
print(f"✅ Results saved to: {dialogue_output_path}")
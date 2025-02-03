import PyPDF2
import re

def convert_pdf_to_text(pdf_path, txt_path):
    """
    Converts a PDF file to a cleaned text file.
    
    - Extracts text from PDF.
    - Fixes formatting issues (newlines, dialogue structure).
    - Saves to a text file.
    
    :param pdf_path: Path to the input PDF file.
    :param txt_path: Path to save the cleaned text file.
    """
    # === Step 1: Extract text from PDF ===
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        raw_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n\n"  # Preserve spacing between pages

    # === Step 2: Clean Extracted Text ===
    cleaned_text = clean_text(raw_text)

    # === Step 3: Save Cleaned Text ===
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Conversion complete! Cleaned text saved to {txt_path}")

def clean_text(text):
    """
    Cleans and reformats text for NLP processing.

    - Fixes newlines breaking chapter titles.
    - Removes metadata, headers, and footers.
    - Formats dialogue with clear breaks.

    :param text: Raw extracted text.
    :return: Cleaned text.
    """
    # Remove metadata (header/footer noise)
    text = re.sub(r"(© Copyright Reserved.*|Pdfcorner.com|Blood Meridian Pdf.*)", "", text, flags=re.IGNORECASE)

    # === Fix Broken First Letters in Chapters ===
    # Example issue: "S\n ee the child." → "See the child."
    text = re.sub(r"\b([A-Z])\n\s*([a-z])", r"\1\2", text)

    # === Format Dialogue ===
    dialogue_patterns = [
        r"(\bsaid\b\s+\w+)",  # Matches "said [character]"
        r"(\bcried\b\s+\w+)",  # Matches "cried [character]"
        r"(\bcalled\b\s+\w+)",  # Matches "called [character]"
        r"(\basked\b\s+\w+)",   # Matches "asked [character]"
        r"(\breplied\b\s+\w+)", # Matches "replied [character]"
    ]

    for pattern in dialogue_patterns:
        text = re.sub(pattern, r"\n\1", text)  # Insert line breaks before common dialogue

    # Ensure clean paragraph spacing
    text = re.sub(r"\n{2,}", "\n\n", text)  # Convert multiple newlines to two

    return text


pdf_path = r"C:/Users\AlexWu\Documents/GitHub/BloodMeridianNLP/Blood-Meridian-Pdf.pdf"  # Name of your PDF file
txt_path = r"C:/Users\AlexWu\Documents/GitHub/BloodMeridianNLP/Blood-Meridian.txt"      # Name of the output text file

# Run the conversion
convert_pdf_to_text(pdf_path, txt_path)
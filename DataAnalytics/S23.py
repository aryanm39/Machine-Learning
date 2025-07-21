import re
def preprocess_text(text):
    # Remove special characters and digits using regular expressions
    processed_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return processed_text

# Sample text paragraph
text_paragraph = """
                    This is a sample text paragraph with some special characters and digits: 12345!@#$%. 
                    We need to preprocess this text to remove these special characters and digits.
                 """
# Preprocess the text
processed_paragraph = preprocess_text(text_paragraph)

print("Original Text:")
print(text_paragraph)
print("\nProcessed Text:")
print(processed_paragraph)
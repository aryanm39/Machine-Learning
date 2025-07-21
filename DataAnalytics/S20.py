import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the stopwords dataset
nltk.download('stopwords')
nltk.download('punkt')

# Input text paragraph
text_paragraph = "Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy."

# Tokenize the text paragraph
words = word_tokenize(text_paragraph)

# Get English stopwords
english_stopwords = set(stopwords.words('english'))

# Remove stopwords from the tokenized words
filtered_words = [word for word in words if word.lower() not in english_stopwords]

# Join the filtered words back into a paragraph
filtered_paragraph = ' '.join(filtered_words)

print("Original Text Paragraph:\n", text_paragraph)
print("\nText Paragraph after Removing Stopwords:\n", filtered_paragraph)
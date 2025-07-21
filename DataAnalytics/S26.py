import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

text_paragraph = """Hello all, Welcome to Python Programming Academy. 
                Python Programming Academy is a nice platform to learn new programming skills. 
                It is difficult to get enrolled in this Academy."""

# Preprocessing: Remove special characters and digits
formatted_text = re.sub(r'[^a-zA-Z\s]', '', text_paragraph)

# Tokenize the text
words = word_tokenize(formatted_text.lower())

# Remove stopwords
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# Calculate word frequencies
word_freq = {}
for word in words:
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

# Compute weighted frequencies
max_freq = max(word_freq.values())
for word in word_freq.keys():
    word_freq[word] = (word_freq[word] / max_freq)

# Calculate sentence scores based on word frequencies
sentence_scores = {}
sentences = sent_tokenize(text_paragraph)
for sentence in sentences:
    for word in word_tokenize(sentence.lower()):
        if word in word_freq:
            if sentence not in sentence_scores:
                sentence_scores[sentence] = word_freq[word]
            else:
                sentence_scores[sentence] += word_freq[word]

# Get top 2 sentences with highest scores
summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)

print("\n---------------------------Summary Text------------------------------\n", summary)

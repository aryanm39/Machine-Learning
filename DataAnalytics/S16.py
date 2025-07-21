import nltk
# nltk.download('all')
nltk.download('stopwords')
nltk.download('punkt')
# Preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

text = """Hello all Students,Welcome to [Today's] Practical. Let's learn How to perform Text"""

# Removing square brackets, digits, and special symbols
text = re.sub(r'[^a-zA-Z\s]', '', text)

# Convert to lowercase
text = text.lower()

print("\n--------------Text after removing digits and special characters-------------------------------\n")
print(text)

# Tokenize the text
words = word_tokenize(text)

print("\n--------------------Extractive Summarization------------------------------------------")
print("\n--------------------Words in text------------------------------------------", words)

# Removing stopwords
stopWords = set(stopwords.words("english"))
words = [word for word in words if word not in stopWords]

# Creating a frequency table of words
wordfreq = {}
for word in words:
    if word in wordfreq:
        wordfreq[word] += 1
    else:
        wordfreq[word] = 1

# Compute the weighted frequencies
maximum_frequency = max(wordfreq.values())

for word in wordfreq.keys():
    wordfreq[word] = (wordfreq[word] / maximum_frequency)

print("\n-----------------Word Frequencies-------------------------\n", wordfreq)

# Creating a dictionary to keep the score of each sentence
sentences = sent_tokenize(text)
sentenceValue = {}

for sentence in sentences:
    for word, freq in wordfreq.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

import heapq

summary = ''
summary_sentences = heapq.nlargest(2, sentenceValue, key=sentenceValue.get)
summary = ' '.join(summary_sentences)
print("\n---------------------------Summary Text------------------------------\n", summary)
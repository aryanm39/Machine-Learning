import nltk
# nltk.download('all') #
nltk.download('stopwords')
nltk.download('punkt')
# Preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

text = """So, keep working. Keep striving. 
            Never give up. 
            Fall down seven times, get up eight. 
            Ease is a greater threat to progress than hardship. 
            Ease is a greater threat to progress than hardship. 
            So, keep moving, keep growing, keep learning. See you at work."""

# Removing special characters and digits
formatted_text = re.sub(r'[^a-zA-Z\s]', '', text)

print("\n--------------Text after removing digits and special characters-------------------------------\n")
print(formatted_text)

# Tokenize the text
words = word_tokenize(formatted_text.lower())

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
sentences = sent_tokenize(formatted_text)
sentenceValue = {}

for sentence in sentences:
    for word, freq in wordfreq.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

# Generate summary
summary = ''
summary_sentences = heapq.nlargest(4, sentenceValue, key=sentenceValue.get)
summary = ' '.join(summary_sentences)
print("\n---------------------------Summary Text------------------------------\n", summary)

#-----------------------------------------------------------------------------------------------------------
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

# Constants
NUM_SENTENCES_IN_SUMMARY = 2

text = """So, keep working. Keep striving. Never give up. Fall down seven times, get up eight. 
Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. 
So, keep moving, keep growing, keep learning. See you at work."""

# Preprocessing
text = re.sub(r'[^\w\s]', '', text)  # Removing special characters
text = re.sub(r'\d+', '', text)       # Removing digits

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

# Generate summary
summary = ''
summary_sentences = heapq.nlargest(NUM_SENTENCES_IN_SUMMARY, sentenceValue, key=sentenceValue.get)
summary = ' '.join(summary_sentences)
print("\n---------------------------Summary Text------------------------------\n", summary)
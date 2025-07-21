import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Textual data to remove stopwords
paragraph_text = """Hello all, Welcome to Python Programming Academy. Python Programming Academy"""

# Sentence Tokenization
sentences = sent_tokenize(paragraph_text)
print("\n-------------Sentences in paragraph---------------------------\n", sentences)

# Word Tokenization and Removing Stopwords
tokenized_words = word_tokenize(paragraph_text.lower())  # Convert to lowercase for consistency
stop_words_data = set(stopwords.words('english'))
filtered_words_list = [word for word in tokenized_words if word not in stop_words_data]

print("\n----------------------Tokenized words----------------------\n", tokenized_words, "\n")
print("\n---------------------Filtered words after removing stopwords-----------------\n",
      filtered_words_list)

# Word Frequency Distribution
frequency_distribution = FreqDist(filtered_words_list)

# Plot word frequency distribution
frequency_distribution.plot(32, cumulative=False)
plt.show()

# Plot word cloud of text
word_cloud = WordCloud(collocations=False, background_color='black').generate(paragraph_text)
plt.figure()
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
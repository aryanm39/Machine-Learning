import nltk
import pandas as pd
data = pd.read_csv('datasets/movie_review.csv')

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader_analyzer = SentimentIntensityAnalyzer()

review1 = "I purchased headphones online. I am very happy with the product."
print("\n----------------Review1---------------\n",vader_analyzer.polarity_scores(review1))

review2 = "I saw the movie yesterday. The animation was really good but the script was ok. "
print("\n----------------Review2---------------\n",vader_analyzer.polarity_scores(review2))

review3 = "I enjoy listening to music."
print("\n----------------Review3---------------\n",vader_analyzer.polarity_scores(review3))

review4 = "I take a walk in the park everyday."
print("\n----------------Review4---------------\n",vader_analyzer.polarity_scores(review4))

#--------------------------------------------------------------------------------------------------------
#Chatgpt-1
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the movie_review.csv dataset
data = pd.read_csv('path_to_your_downloaded_movie_review.csv')

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Define sample reviews
reviews = data['text']

# Analyze sentiment of each review
sentiments = [vader_analyzer.polarity_scores(review) for review in reviews]

# Create a word cloud from the reviews
text = ' '.join(data['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

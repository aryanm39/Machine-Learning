#Importing Libraries
import pandas as pd 
import numpy as np 

#Reading Dataset
data = pd.read_csv("datasets/Covid_2021_1.csv")
df=pd.DataFrame(data)

#Data Cleaning operation
df.dropna()

print("Displaying Dataset:\n",df.head(5))

#Tokenize comments in words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Converting columns data to string
comment_text = df['comment_text'].values.astype(str)
comments=np.array_str(comment_text)
tokenized_words=word_tokenize(comments)
print("\n-----------------Comments into Tokenized words are: \n",tokenized_words)

#Perform sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#create an object of sentimentIntensityAnalyzer()
vader_analyzer=SentimentIntensityAnalyzer()
total=len(comment_text)
pos_cnt=0
neg_cnt=0
neu_cnt=0
for i in comment_text:
    result1 = vader_analyzer.polarity_scores(i)
    if result1['compound']>=0.05:
        pos_cnt=pos_cnt+1
    elif result1['compound']<=-0.05:
        neg_cnt=neg_cnt+1
    else:
        neu_cnt=neu_cnt+1

#Display percentage of positive,negative and neutral comments
print("\n Percentage of positive Comments : \n",(pos_cnt/total)*100,"%")
print("\n Percentage of Negative Comments : \n",(neg_cnt/total)*100,"%")
print("\n Percentage of neutral Comments : \n",(neu_cnt/total)*100,"%")


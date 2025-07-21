import pandas as pd
import matplotlib.pyplot as plt

# reading dataset
data = pd.read_csv("datasets/INvideos.csv")
data.dropna()

# convert 'publish_time' to datetime format
data['publish_time'] = pd.to_datetime(data['publish_time'])
# Find the total views, total likes, total dislikes, and comment count
print("\nThe total views, total likes, total dislikes, and comment count:")
print("Total Views:",data['views'].sum())
print("Total Likes:", data['likes'].sum())
print("Total Dislikes:",data['dislikes'].sum())
print("Total Comments:", data['comment_count'].sum())

#Find the least and top most liked and commneted videos
print("\nLeast Liked video: ",data.sort_values(by='likes').head(1))
print("\nLeast Commented video: ",data.sort_values(by='comment_count').head(1))
print("\nTopmost Liked video: ",data.sort_values(by='likes',ascending=False).head(1))
print("\nTopmost Commented video: ",data.sort_values(by='comment_count',ascending=False).head(1))

# Perform year-wise statistics for views and plot the analyzed data
data['year'] = data['publish_time'].dt.year
yearly_views = data.groupby('year')['views'].sum()
plt.plot(yearly_views.index, yearly_views.values, marker='o')
plt.xlabel("Year")
plt.ylabel("Views")
plt.title("Year-wise Views")
plt.show()

# Plot the viewers who reacted on videos
reacted = data['likes'].sum() + data['dislikes'].sum()
total_views = data['views'].sum()
neutral = total_views - reacted
labels = ['Reactors', 'Neutral']
values = [reacted, neutral]
plt.pie(values, labels=labels, autopct='%1.2f%%')
plt.title("Viewers who reacted on videos")
plt.show()
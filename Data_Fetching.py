import pandas as pd
import numpy as np
import requests



df = pd.read_csv("dialect_dataset.csv")
URL = "https://recruitment.aimtechnologies.co/ai-tasks"
tweets = []
ids = list(df["id"].astype(str))
ranges = np.arange(0, len(ids), 1000).tolist()
ranges.append(ranges[-1]+(len(ids)%1000))


for i in range(len(ranges)-1):
    data = ids[ranges[i]:ranges[i+1]]
    tweet = requests.post(URL, json=data).json()
    print(tweet.values())
    tweets.extend(tweet.values())


# save to CSV
df.insert(1, "tweet", tweets, True)
df.to_csv(r'tweet_dialect_dataset.csv')


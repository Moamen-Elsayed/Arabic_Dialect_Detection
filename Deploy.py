import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
import arabicstopwords.arabicstopwords as stp
import re


def text_normalization(text):
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel,"", text)
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى",
              "\\", '\n', '\t', '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا",
               "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ']
    text = re.sub("#", ' ', text)
    text = text.replace('_', ' ')
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"[a-zA-Z]", '', text)
    text = re.sub(r"\d+", ' ', text)
    text = re.sub(r"\n+", ' ', text)
    text = re.sub(r"\t+", ' ', text)
    text = re.sub(r"\r+", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    text_cleaning_re = "@\S+|https?:\S+|http?:\S"
    text = re.sub(text_cleaning_re, ' ', str(text)).strip()
    return text

def remove_stopWords(sentence):
    terms=[]
    stopWords= set(stp.stopwords_list())
    for term in sentence.split() : 
        if term not in stopWords :
            terms.append(term)
    return " ".join(terms)


class Tweet(BaseModel):
    tweet: str

app = FastAPI()

with open(r'Models/MultinomialNB.pkl', 'rb') as f:
    loaded_count_vectorizer, loaded_clf = pickle.load(f)



@app.get('/')
def root():
    return {'message': 'تصنيف اللهجات العربية'}

@app.post('/classify')
def classify_dialect(tweet: Tweet):
    tweet = tweet.dict()
    tweet = tweet["tweet"]
    cleaned_tweet = text_normalization(tweet)
    cleaned_tweet = remove_stopWords(cleaned_tweet)
    pred = loaded_clf.predict(loaded_count_vectorizer.transform([cleaned_tweet]))[0]

    return {
        'اللهحة هي': pred
    }

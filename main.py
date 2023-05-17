import streamlit as st
import sys
import pickle
from nltk.corpus import stopwords
import regex as re
import sklearn
import numpy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import wordnet
import threading

stop_words = stopwords.words('english')
new_stopwords = ["movie","one","film","would","shall","could","might"]
stop_words.extend(new_stopwords)
stop_words.remove("not")
stop_words=set(stop_words)

# lemmatization of word
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]


class ProcessMetaThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)


# Removing special character
def remove_special_character(content):
    return re.sub('\W+', ' ', content)  # re.sub('\[[^&@#!]]*\]', '', content)


# Removing URL's
def remove_url(content):
    return re.sub(r'http\S+', '', content)


# Removing the stopwords from text
def remove_stopwords(content):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)


# Expansion of english contractions
def contraction_expansion(content):
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"n\'t", " not", content)
    content = re.sub(r"\'re", " are", content)
    content = re.sub(r"\'s", " is", content)
    content = re.sub(r"\'d", " would", content)
    content = re.sub(r"\'ll", " will", content)
    content = re.sub(r"\'t", " not", content)
    content = re.sub(r"\'ve", " have", content)
    content = re.sub(r"\'m", " am", content)
    return content


# Data preprocessing
def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_special_character(content)
    content = remove_url(content)
    content = remove_stopwords(content)

    return content

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('modelLogReg.pkl','rb'))

st.title("Sentiment Analyzer")

input_com = st.text_input("Enter a comment")


if st.button('Predict'):

    input_com = str(input_com)
    # 1. preprocess
    transformed_com = data_cleaning(input_com)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_com])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == '1':
        st.header("Positive Sentiment \U0001F60A")
    else:
        st.header("Negative Sentiment \U00002639")




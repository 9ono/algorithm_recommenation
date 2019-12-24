
from flask import Flask, render_template, request
app = Flask(__name__)
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import re
import os.path
import math
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from scipy import sparse
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

mydata_train = pd.read_csv('./Data/Prepossed/Prepossed_data_train_shuffled.csv')
mydata_test = pd.read_csv('./Data/Prepossed/Prepossed_data_test_Shuffled.csv')
mydata =  pd.read_csv('./Data/Prepossed/Prepossed_data.csv')
mydata = mydata.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis = 1)
mydata_train = mydata_train.drop(['Unnamed: 0.1'], axis = 1)
mydata_test = mydata_test.drop(['Unnamed: 0.1'], axis = 1)

train_X, train_y = mydata_train['Plot'], mydata_train.drop(['BID', 'Plot'], axis=1)
test_X, test_y = mydata_test['Plot'], mydata_test.drop(['BID', 'Plot'], axis=1)

category_columns = train_y.columns

#plt.style.use('fivethirtyeight')
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

from helper_functions import *
overall_f1_score_v1_cv = make_scorer(overall_f1_score_v1, greater_is_better=True)
prob_thresh = get_prob_thresh(mydata[category_columns], thresh_sel=2)
pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])
parameters = {
            'tfidf__max_df': [0.25, 0.5, 0.75],
            'tfidf__min_df': [1, 2],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            "clf__estimator__C": [0.1, 1],
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
grid_search_cv.fit(train_X, train_y)

# measuring performance on test set
best_clf = grid_search_cv.best_estimator_
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
n=WordNetLemmatizer()

def prep(text):
    
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    text = shortword.sub('', text)
    text = re.sub(r'[?|!|\'|"|#|_]', '', text)
    text = re.sub(r'[,|.|;|:|(|)|{|}|\|/|<|>]|-', ' ', text)
    text = text.replace("\n"," ")
    text = re.sub('[^a-z A-Z]+', ' ', text)
    text = text.lower()
    
    
    text = tokenizer.tokenize(text)
    text = [n.lemmatize(w) for w in text]
    
    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten',
                       'may','also','across','among','beside','however','yet','within',
                      'integer', 'number', 'contain', 'line', 'first'])
    result = []
    for w in text: 
        if w not in stop_words: 
            result.append(w) 
    text = ' '.join(result)
    
    return text

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method =='GET':
        return render_template("answer.html")

    if request.method == 'POST':
        hi = request.form
        y_pred = pd.DataFrame(columns=category_columns)
        inputs = hi['contents']
        inputs = prep(inputs)

        jojo = pd.DataFrame([inputs],columns = ['input'])

        prob = best_clf.predict_proba(jojo['input'])

        for idx, col in enumerate(category_columns):
                y_pred[col] = prob[:,idx]>prob_thresh[idx]
        foo = []
        for i,col in zip(y_pred.loc[0],y_pred.columns):
            if i==True:
                foo.append(col)

        return render_template("index.html", jj = hi, foo=foo)
        
if __name__ == '__main__':
    app.run()
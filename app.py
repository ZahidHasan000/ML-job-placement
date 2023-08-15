from flask import Flask
#from flask_restful import API
app = Flask(__name__)
#api= app(API)

# import sklearn
# from sklearn.decomposition import TranscatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
list_sentences=[
    "This is the first the document.",
    "This document is the second document",
    "And this is the third one",
    "Is this is the first document?",
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(list_sentences)
print(vectorizer.get_feature_names_out())
print(X.shape)

import pandas as pd
import os

data_file_path = os.path.join('data', 'Salary_Data.csv')
dataset = pd.read_csv(data_file_path, encoding = 'unicode_escape')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(y_pred)

@app.route('/')
def home():
    return('Hello flask world')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="8080",debug="true")
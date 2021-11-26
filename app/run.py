import json
import plotly
import re
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine


app = Flask(__name__)

#TOOD: The RegEx constants and all the logic related to both the class UrgencyWordExtractor 
#      as well as the tokenize function should be part of the data package that we import above
#      that way we are not duplicating this code here, but since Udacity did it for the tokenize 
#      function then we will also take this shortcut for now and fix at a later time.

# Regular expression to find URLs in text
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# Regular expression to find alpha numeric characters to isolate punctuation
alphanum_regex = "[^a-zA-Z0-9]"

# From Synonyms of EMERGENCY by Oxford Dictionary on Lexico at https://www.lexico.com/synonyms/emergency
urgent_words = ['emergency', 'crisis', 'urgent', 'extremity', 'exigency', 'accident', 'catastrophe', 'calamity'
                , 'difficulty', 'plight', 'predicament', 'tight spot', 'tight corner', 'mess', 'quandary', 'dilemma'
                , 'unforeseen circumstances', 'desperate straits', 'dire straits', 'danger', 'critical']

class UrgencyWordExtractor(BaseEstimator, TransformerMixin):
    """
    A class to find instances of synonyms for the word EMERGENCY.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    urgent_words(text):
        Determines if the given text contains any of the known synonyms for emergency.
   fit(x, y=None):
        Returns default model fit.
   transform(X):
        Returns a Pandas DataFrame where urgent_words was applied.
    """
    def urgent_words(self, text):
        """
        Determines if the given text contains any of the known synonyms for emergency.

        Parameters
        ----------
        text : str
            The string text to analyze

        Returns
        -------
        True if the given string text has any synonyms for emergency, False otherwise.
        """
        if any(word in urgent_words for word in text.lower()):
            return True
        return False

    def fit(self, x, y=None):
        """
        Returns default model fit.

        Parameters
        ----------
        x : Pandas.Series
            x values
        y : Pandas.Series, default None
            y values

        Returns
        -------
        Returns self.
        """
        return self

    def transform(self, X):
        """
        Transforms the given Pandas Series X into a Pandas DataFrame with the urgent_words filter applied.

        Parameters
        ----------
        X : Pandas.Series
            X values

        Returns
        -------
        Pandas DataFrame.
        """
        X_tagged = pd.Series(X).apply(self.urgent_words)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    """
    Tokenize and clean up the given text using NLTK to Lemmatize and clean to lower case as well as strip white space, remove stop words and any URLs

    Parameters
    ----------
    text : string
        The text string we are tokenizing and cleaning

    Returns
    -------
    array
        Return the cleaned string tokens in a string array

    Examples
    --------
    >>> tokens = tokenize("Hello, world, are you there?")
    
    """
    # Let's clean up any URLs in the text so they will not cause any unwanted noise in the model
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Initialize the Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(alphanum_regex, " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    This is the index webpage which displays visuals by way of Flask and the jinja2 engine to receive user input text and run against the model

    Parameters
    ----------
    None

    Returns
    -------
    string
        Return the generated HTML for this page

    """    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create a percentage count of all categories that are represented in the data
    category_counts = []
    category_names = []
    total_count = 0
    for column in df.columns:
        if (column not in ['id', 'message', 'original', 'genre']):
            category_count = df.loc[df[column] != 0, column].sum()
            if (category_count > 0):
                category_counts.append(category_count)
                category_names.append(column.replace("_", " "))
                total_count += category_count
    category_counts = [acount / total_count for acount in category_counts]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=category_names,
                    values=category_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Represented Categories',
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    This is the webpage that handles user input query and displays model results

    Parameters
    ----------
    None

    Returns
    -------
    string
        Return the generated HTML for this page

    """    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    The main function of this web application dashboard script

    Parameters
    ----------
    None

    Returns
    -------
    None

    """    
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

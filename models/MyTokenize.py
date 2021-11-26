import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Regular expression to find URLs in text
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# Regular expression to find alpha numeric characters to isolate punctuation
alphanum_regex = "[^a-zA-Z0-9]"

def MyTokenize(text):
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
    >>> tokens = MyTokenize("Hello, world, are you there?")
    
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

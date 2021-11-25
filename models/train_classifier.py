import sys
import pickle
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

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
        if any(word in urgent_words for word in text):
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
    
def load_data(database_filepath):
    """
    Load the messages data from a SQL table using the given database_filename SQL connection

    Parameters
    ----------
    database_filepath : string
        The file path of the database SQL connection

    Returns
    -------
    pandas.Series
        Return the X input mesages
    pandas.Series
        Return the Y output categories
    string
        Return the category names

    Examples
    --------
    >>> X, Y, category_names = load_data(database_filepath)
    
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X, y = make_multilabel_classification(n_classes=36, random_state=0)
    X = df.message.values
    
    df.apply(pd.to_numeric, errors='ignore')
    # y Needs to be an array of arrays where each inner array is 36 elements, where our binary categories start after column 3
    for i in range(0, len(y)):
        for j in range(0, len(y[i])):
            y[i, j] = df.iloc[i][j+3]
    #y = df.loc[:, ~df.columns.isin(['message', 'original', 'genre'])]
    
    return X, y, y.columns

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
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Return the ML pipeline that will take in the message column as input and output classification results on the other 36 categories in the dataset

    Parameters
    ----------
    None

    Returns
    -------
    sklearn.pipeline.Pipeline
        Return the Machine Learning Pipeline that will process our text and run the ML model

    Examples
    --------
    >>> model = build_model()
    
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('urgent_words', UrgencyWordExtractor())
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier())),
    ])
    
    # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4],
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the given model Pipeline with the given X test data and expected Y test data

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The given model Pipeline to evaluate
    X_test: pandas.Series
        The X test input mesages
    Y_test: pandas.Series
        The Y test output categories
    category_names: string
        The list of category names

    Returns
    -------
    None

    Examples
    --------
    >>> evaluate_model(model, X_test, Y_test, category_names)
    
    """
    #print("    MODEL SCORE: {}".format(model.score(X_test, Y_test)))
    #labels = np.unique(y_pred)
    y_pred = model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, y_pred, labels=category_names)
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)

def save_model(model, model_filepath):
    """
    Saves the results of the given model Pipeline to the pickle file path given

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The given model Pipeline whose results we are saving
    model_filepath : string
        The file path where we write the model results as a pickle file

    Returns
    -------
    None

    Examples
    --------
    >>> save_model(model, model_filepath)
    
    """
    # open the model_filepath for writing in binary mode
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """
    The main function of this ML pipeline script. If we are not passed the expected 2 parameters we simply return a message stating the parameters we need, otherwise we run the ML pipeline process

    Parameters
    ----------
    database_filepath : string
        The file path to the database connection file
    model_filepath : string
        The file path where we write the model results as a pickle file

    Returns
    -------
    None

    Examples
    --------
    >>> python models/train_classifier.py
    Please provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file to save the model to as the second argument.

    Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    
    >>> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    Loading data...
        DATABASE: data/DisasterResponse.db
    Building model...
    Training model...
    Evaluating model...
        MODEL SCORE: 0.8675309
    Saving model...
        MODEL: models/classifier.pkl
    Trained model saved!
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

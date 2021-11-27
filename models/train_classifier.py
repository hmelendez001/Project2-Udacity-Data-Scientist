import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sklearn.metrics as metrics

from UrgencyWordExtractor import UrgencyWordExtractor
from MyTokenize import MyTokenize

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
    # Create a separate DataFrame with just the Y categories by dropping message, original, and genre
    df_categories = df.drop(['message', 'original', 'genre'], axis = 1)
    # Use this helper function to create the shape of our X, y accordingly. We have to do this because 
    # we have multiple y values we are predicting not just one value
    X, y = make_multilabel_classification(n_features=1, n_classes=len(df_categories.columns), random_state=0)
    category_names = df_categories.columns
    # Fill y with our data category values instead of the ones make_multilabel_classification generated
    for i in range(0, len(y)):
        for j in range(0, len(y[i])):
            y[i, j] = df_categories.iloc[i][j]
    # Initially we thought X had to be exactly as make_multilabel_classification generated which is a 1-d array of arrays, 
    # which is why originally we did X.append([]) (now commented out) then X[k].append(df.message[k]), but then our Pipeline errors showed 
    # it wanted just a simply 1-d array for X.
    lenX = len(X)
    X = [] * lenX
    for k in range(0, lenX):
        ####X.append([])
        X.append(df.message[k])
    
    return X, y, category_names

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
    # We are using the FeatureUnion to put together our Pipeine which allows us to 
    # run various transformations as well as a check in parallel. Also note that 
    # our classifier has to be a MultiOutputClassifier because our output is predicting 
    # across multiple values or categories
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=MyTokenize)),
                ('tfidf', TfidfTransformer()),
            ])),

            ('urgent_words', UrgencyWordExtractor()),
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier())), # Giving us Weighted Precision: 0.67
        #('clf', MultiOutputClassifier(SGDClassifier(loss='log', random_state=1, max_iter=5))), # ValueError: The number of class labels must be greater than one.
        #('clf', MultiOutputClassifier(estimator=SVC(C=1.0, cache_size=200, # ValueError: The number of class labels must be greater than one.
        #                                            class_weight=None, coef0=0.0,
        #                                            decision_function_shape='ovr', degree=3,
        #                                            gamma=0.001, kernel='rbf', max_iter=-1,
        #                                            probability=False, random_state=None,
        #                                            shrinking=True, tol=0.001, verbose=False))),    
    ])
    
    # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__smooth_idf': (True, False), ### Best is True the default
        #'features__text_pipeline__tfidf__sublinear_tf': (True, False), ### Best is True the default
        'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__n_neighbors': (2, 5, 10), ### Best is 2 the default
        #'clf__n_estimators': [50, 100, 200],
        #'clf__min_samples_split': [2, 3, 4],
        #'features__transformer_weights': ( ##TypeError: no supported conversion for types: (dtype('float64'), dtype('O'))
        #    {'text_pipeline': 1, 'urgent_words': 0.5},
        #    {'text_pipeline': 0.5, 'urgent_words': 1},
        #    {'text_pipeline': 0.8, 'urgent_words': 1},
        #),
        }

    # create grid search object in order to find the best parameters based on the features above
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
    # run the model predictions to evaluate performance
    y_pred = model.predict(X_test)
    # For confusion matrix we need the .argmax(axis=1) argument because it must be a list of predictions, not OHEs (one hot encodings)
    # Thanks to user cs95 on StackOverFlow for this tip found at: https://stackoverflow.com/a/46954067/2788414
    confusion_mat = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))#, labels=category_names)
    
    # Shout out to Joydwip Mohajon the author of the article "Confusion Matrix for Your Multi-Class Machine Learning Model" where 
    # he shows an example of displaying accuracy_score, precision_score, recall_score, f1_score from sklearn.metrics
    # found at https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
    print("    Labels:", category_names)
    print("    Confusion Matrix:\n", confusion_mat)
    ##print("    Accuracy:", (y_pred == Y_test).mean())
    print("\n    Best Parameters:", model.best_params_)
    
    print('\n    Accuracy: {:.2f}\n'.format(accuracy_score(Y_test, y_pred)))
    # We are seeing runtime warnings from scikit 
    # (UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples)
    # According to user Mohsenasm on StackOverFlow this means that there is no F-score to calculate for this label (no predicted samples), 
    # and thus the F-score for this case is considered to be 0.0. Since we requested an average of the score, we must take into account 
    # that a score of 0 was included in the calculation, and this is why scikit-learn is showing us that warning.
    # see https://stackoverflow.com/a/47285662/2788414
    # This was our first clue that the dataset we were given was imbalanced. There is no way to impute missing text data at this time. 
    # This is why we introduced the categories array: non_missing_categories, so that we could get a "Balanced" precision, recall, f1-score 
    # averages below using these labels/categories
    non_missing_categories = ['related', 'aid_related', 'weather_related', 'direct_report', 'request', 'other_aid', 'food', 'earthquake', 'storm'
                              , 'shelter', 'floods', 'medical_help', 'infrastructure_related', 'water', 'other_weather', 'buildings', 'medical_products'
                             ]

    print('    Weighted Precision: {:.2f}'.format(precision_score(Y_test, y_pred, average='weighted')))
    print('    Weighted Recall: {:.2f}'.format(recall_score(Y_test, y_pred, average='weighted')))
    print('    Weighted F1-score: {:.2f}'.format(f1_score(Y_test, y_pred, average='weighted')))

    # We can see the categories with 0.00 precision, recall, f1-score, support that are affecting our overall numbers, 
    # For example, offer, military, child alone (this explains why tests on our UI web app with 'child lost' or 'child alone' 
    # are not coming up with a child alone positive). This is why we introduced target_names=non_missing_categories instead of 
    # using all the categories with target_names=category_names
    print('\nClassification Report\n')
    print(classification_report(Y_test, y_pred, target_names=non_missing_categories))

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
        Labels: Index(['related', 'request', 'offer', 'aid_related', 'medical_help',
           'medical_products', 'search_and_rescue', 'security', 'military',
           'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
           'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report'],
          dtype='object')
        Confusion Matrix:
     [[20]]
        
        Best Parameters: {'features__text_pipeline__tfidf__use_idf': True, 'features__text_pipeline__vect__max_df': 0.5, 'features__text_pipeline__vect__max_features': None, 'features__text_pipeline__vect__ngram_range': (1, 2)}

        Accuracy: 0.15
        Weighted Precision: 0.61
        Weighted Recall: 0.80
        Weighted F1-score: 0.69
        
        Classification Report
                        precision    recall  f1-score   support

               related       1.00      0.84      0.91        19
           aid_related       0.77      0.91      0.83        11
       weather_related       0.00      0.00      0.00         0
         direct_report       0.59      0.91      0.71        11
               request       0.00      0.00      0.00         1
             other_aid       0.00      0.00      0.00         1
                  food       0.00      0.00      0.00         0
            earthquake       0.00      0.00      0.00         1
                 storm       0.00      0.00      0.00         0
               shelter       0.00      0.00      0.00         0
                floods       0.50      0.75      0.60         4
          medical_help       0.83      0.83      0.83         6
infrastructure_related       0.00      0.00      0.00         1
                 water       0.00      0.00      0.00         0
         other_weather       0.00      0.00      0.00         0
             buildings       0.00      0.00      0.00         0
      medical_products       0.00      0.00      0.00         0

           avg / total       0.61      0.80      0.69        64
    
   Saving model...
        MODEL: models/classifier.pkl
    Trained model saved!
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # Set up our model training with only test_size percent of the data. This is where the machine learning 
        # part of the Pipeline is "given the answers to the test" for part of the data or the test_size. 
        # Later on we can run the remaining 1 - test_size of the data to test how well we actually 
        # perform against "real data."
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        # This is the actual training of the model against test_size of the dataset
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # This is the part where we run against the remaining 1 - test_size of the dataset to see how well we do
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # Now we save the model so we can later load it and re-run it from our web application dashboard on demand based on new user input
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

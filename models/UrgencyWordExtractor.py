import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

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

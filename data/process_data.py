import sys
import pandas as pd
from pandas.api.types import is_string_dtype
from sqlalchemy import create_engine, select, MetaData
from sqlalchemy.ext.declarative import declarative_base

def load_data(messages_filepath, categories_filepath):
    """
    Take the path to a file with online messages and a the path to a file with corresponding categories and merge the two into one merged Pandas DataFrame.

    Parameters
    ----------
    messages_filepath : string
        The file path of the file containing the raw messages
    categories_filepath : string
        The file path of the file containing the raw message categories

    Returns
    -------
    pandas.DataFrame
        Return a merged Pandas DataFrame dataset

    See Also
    --------
    clean_data : Clean a given merged DataFrame dataset.

    Examples
    --------
    >>> df = load_data(messages_filepath, categories_filepath)
    
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # create a dataframe of the 36 individual category columns
    # I stored the new 36 columns into into its own DataFrame categories36 because 
    # I want to preserve the original id column in categories so I can use it to merge later
    categories36 = categories.categories.str.split(";", expand=True)
    # add back the id column because we will need it to merge the two datasets by
    categories36['id'] = categories['id']

    # select the first row of the categories dataframe
    row = categories36.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to replace all instances of -0 and -1 with a blank string 
    # so we are left with only the name
    category_colnames = [row[x].values[0].replace("-0","").replace("-1","") for x in range(0, len(categories36.columns) - 1)]
    # let's not forget the id column, we'll need that again for merging
    category_colnames.append('id')

    # rename the columns of `categories`
    categories36.columns = category_colnames
    
    # Before attempting a merge via the concatenate call, clean up duplicates, if any
    if messages.duplicated(subset=['id']).sum() > 0:
        print("Cleaning up messages duplicates {}...".format(messages.duplicated(subset=['id']).sum()))
        messages.drop_duplicates(inplace=True, subset=['id'])
    if categories36.duplicated(subset=['id']).sum() > 0:
        print("Cleaning up categories duplicates {}...".format(categories36.duplicated(subset=['id']).sum()))
        categories36.drop_duplicates(inplace=True, subset=['id'])
    
    # concatenate the original dataframe with the new `categories36` dataframe by id
    frames = [messages.set_index('id'), categories36.set_index('id')]
    df = pd.concat(frames, sort=True, axis=1, join='inner')
    
    return df

def clean_data(df):
    """
    Given a Pandas DataFrame make sure the column names do not have trailing -0 or -1, that each column that can be numeric is converted to numeric, and that we drop any duplicate rows, if any

    Parameters
    ----------
    df : pandas.DataFrame
        The Pandas DataFrame that we need to clean up

    Returns
    -------
    pandas.DataFrame
        Return the cleaned up Pandas DataFrame dataset

    See Also
    --------
    load_data : Take a messages and a categories file path and load both datasets into one merged Pandas DataFrame.

    Examples
    --------
    >>> df = clean_data(df)
    
    """
    # Convert category values to just numbers 0 or 1, they are still strings
    for column in df.columns:
        if is_string_dtype(df[column]) and (df[column].iloc[0].endswith("-0") or df[column].iloc[0].endswith("-1")):
            # set each value to be the last character of the string
            df[column] = df[column].apply(lambda x: x.replace("{}-".format(column), ""))
            # convert column from string to numeric
            df[column] = df[column].astype(str).astype(int)
            # Let's make sure our values are binary, so either 0 or 1 (if not 0)
            df[column] = df[column].apply(lambda x: 1 if x != 0 else 0)

    return df

def save_data(df, database_filename):
    """
    Save the given Pandas DataFrame to the given database_filename SQL connection database

    Parameters
    ----------
    df : DataFrame
        The Pandas DataFrame that we need to save to the database
    database_filename : string
        The file path of the database SQL connection

    Returns
    -------
    None

    See Also
    --------
    load_data : Take a messages and a categories file path and load both datasets into one merged Pandas DataFrame.

    Examples
    --------
    >>> save_data(df, database_filepath)
    
    """
    # Initialize the SQL engine connection given the database_filename
    engine = create_engine('sqlite:///' + database_filename)
    # Write the DataFrame to SQL, added if_exists = 'replace' to make this call idempotent thanks to the udacity reviewer for this tip
    df.to_sql('Messages', engine, index=False, if_exists = 'replace')  

def main():
    """
    The main function of this ETL data load script. If we are not passed the expected 3 parameters we simply return a message stating the parameters we need, otherwise we run the ETL process

    Parameters
    ----------
    messages_filepath : string
        The file path to the messages raw data CSV file
    
    categories_filepath : string
        The file path to the categories raw data CSV file
    
    database_filepath : string
        The file path to the database connection file

    Returns
    -------
    None

    Examples
    --------
    >>> python data/process_data.py
    Please provide the filepaths of the messages and categories datasets as the first and second argument respectively, 
    as well as the filepath of the database to save the cleaned data to as the third argument. 

    Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    
    >>> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    Loading data...
        MESSAGES: data/disaster_messages.csv
        CATEGORIES: data/disaster_categories.csv
    Cleaning data...
    Saving data...
        DATABASE: data/DisasterResponse.db
    Cleaned data saved to database!
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

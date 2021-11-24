# Project2-Udacity-Data-Scientist
As part of the udacity.com Data Scientist nanodegree, this is project two of the program. This project involves a Disaster Response Pipeline dashboard web application showcasing the following Data Science and Software Engineering Best Practices:

* Github and Code Quality including this repository along with comments, docstring in each function, class, or method, unit tests, and logical functions and PEP8 style guideline conventions
* ETL or Extract Transform Load data setup of a clean dataset
* Machine Learning including NLP techniques to process text data and the proper use of pipelines and grid search, traning vs. test data, and model evaluation
* Deployment of our web application showing our Disaster Response Pipeline visualizations

As in my previous project, I have also documented the work in the blog below:

#### My BLOG is HERE: https://hmelendez001.github.io/2021/11/19/Udacity-Data-Scientist-Nanodegree-Project-2.html

# How to Run the Python Scripts
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

# How to Run the Web Application
python run.py

# Libraries Used
| Library | Description |
| :--- | :--- |
| Bootstrap | This toolkit was used to simplify writing the HTML5 elements including CSS styles in our web application |
| Flask | This Python micro web framework was used to write the backend of our web application dashboard |
| Gunicorn | This is needed for web application deployment |
| Heroku | This is used to access the hosting platform |
| NLTK | This is used by the machine learning pipeline to do the text processing |
| Numpy | This is where numerical constants like np.nan came from|
| Pandas | This is the work horse behind our analysis for reading in the DataFrame from the CSV file and performing various data analysis and preparation |
| Pickle | 
| Plotly | This was the graphing library used to generate our visualizations |
| Sklearn | This is the scikit-learn's Pipeline and GridSearchCV used to build and run our machine learning model |
| Sqlite3 | This is the SQLite database package used to write to the database |

# Files in this Repository
| File | Description |
| :--- | :--- |
| - app | The dashboard or web application main directory |
| - app - master.html | The main page or the landing of the web application |
| - app - go.html | Page which displays the classification results of the web application |
| - app - run.py | The Flask file that runs the web application |
| - data | The directory contaning the raw data for this project along with the scripts to run the ETL pipeline |
| - data - disaster_categories.csv | The raw data containing the disaster organization categories to process |
| - data - disaster_messages.csv | The raw online messages to process |
| - data - process_data.py | A data cleaning pipeline that loads the messages and categories datasets from the raw data, merges the two datasets, cleans the data, and stores it in a SQLite database using an SQLAlchemy engine |
| - data - InsertDatabaseName.db | The SQLite3 database to save the clean data |
| - models | The directory contaning the scripts to run the Text Processing, Natural Language Processing (NLP), and Machine Learning (ML) pipelines |
| - models - train_classifier.py | A machine learning pipeline that loads data from a SQLite database, splits the dataset into training and test datasets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, and exports the final model as a pickle file |
| - models - classifier.pkl | The saved model pickle file |
| Procfile | Instructs the runtime to use gunicorn to run our dashboard |
| README.md | The file you are currently reading |
| requirements.txt | Contains the list of required libraries as listed in the "Libraries Used" section above but includes the versions required at run time |
| runtime.txt | The Python runtime version being used |

# Summary of the results
TODO: The dataset was imbalanced (i.e. some labels like water have few examples). This imbalance affected training the model...TODO..., and your thoughts about emphasizing precision or recall for the various categories...TODO...

# Acknowledgements
Several code snippets came from previous lessons in our Udacity Data Scientist program. Also, where employed I have credited various contributors from StackOverflow.com, kite.com, and the Data Science Stack Exchange at https://datascience.stackexchange.com. A big thank you to our instructors and all those involved in the Udacity program.

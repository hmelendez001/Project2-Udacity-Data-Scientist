# Project2-Udacity-Data-Scientist
As part of the udacity.com Data Scientist nanodegree, this is project two of the program. Following a disaster, there are millions of tweets, news alerts, or social media and online messages that are generated. During this critical time, organizations are overwhelmed with data and they need to filter out what is pertinent information from what is just noise. One response organization might be interested in information about water needs, or roads, or medical supplies, etc. But is someone just commenting on water or urgently asking for clean water needs without using the word water? 

There is not enough time for a person to parse through so much data in time for the correct organization to respond effectively. One in every 1,000 messages might actually be important.

Figure Eight, a company specializing in AI data solutions, has provided real social media and online messages that were generated in an actual emergency that will be used for this project.

This project involves a Disaster Response Pipeline dashboard web application showcasing the following Data Science and Software Engineering Best Practices:

* Github and Code Quality including this repository along with comments, docstring in each function, class, or method, unit tests, and logical functions and PEP8 style guideline conventions
* ETL or Extract Transform Load data setup of a clean dataset
* Machine Learning including NLP techniques to process text data and the proper use of pipelines and grid search, traning vs. test data, and model evaluation
* Deployment of our web application showing our Disaster Response Pipeline visualizations

As in my previous project, I have also documented the work in the blog below:

#### My BLOG is HERE: https://hmelendez001.github.io/2021/11/19/Udacity-Data-Scientist-Nanodegree-Project-2.html

# How to Run the Python Scripts
From the root folder run the following python command to process and clean the raw data:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
Altenatively, to capture run times, from the root folder run the following python command with the time -v prefix:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; time -v python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

From the root folder run the following python command to train and save the model:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Altenatively, to capture run times, from the root folder run the following python command with the time -v prefix:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; time -v python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# How to Run the Web Application
From the app folder run the following python command:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; python run.py

# Libraries Used
| Library | Description |
| :--- | :--- |
| Bootstrap | This toolkit was used to simplify writing the HTML5 elements including CSS styles in our web application |
| Flask | This Python micro web framework was used to write the backend of our web application dashboard |
| Gunicorn | This is needed for web application deployment |
| Heroku | This is used to access the hosting platform |
| JQuery | This is the javascript for manipulating objects on the UI web application dashboard |
| Json | This is the package for manipulating JSON text and objects |
| NLTK | This is used by the machine learning pipeline to do the text processing |
| Numpy | This is where numerical constants like np.nan came from|
| Pandas | This is the work horse behind our analysis for reading in the DataFrame from the CSV file and performing various data analysis and preparation |
| Pickle | This is the utility used to store the trained and optimized model for reuse by the UI web application dashboard |
| Plotly | This was the graphing library used to generate our visualizations |
| Re | This is the Regular Expression package used for manipulating strings with regex |
| Sklearn | This is the scikit-learn's Pipeline and GridSearchCV used to build and run our machine learning model |
| Sqlalchemy | This is the SQLite database package used to write and read to and from the SQL database |

# Files in this Repository
| File | Description |
| :--- | :--- |
| app | The dashboard or web application main directory |
| app > templates | Directory with the HTML files for the web application |
| app > templates > go.html | Page which displays the classification results of the web application |
| app > templates > master.html | The main page or the landing of the web application |
| app > run.py | The Flask file that runs the web application |
| data | The directory contaning the raw data for this project along with the scripts to run the ETL pipeline |
| data > DisasterResponse.db | The SQL Lite connection file |
| data > disaster_categories.csv | The raw data containing the disaster organization categories to process |
| data > disaster_messages.csv | The raw online messages to process |
| data > process_data.py | A data cleaning pipeline that loads the messages and categories datasets from the raw data, merges the two datasets, cleans the data, and stores it in a SQLite database using an SQLAlchemy engine |
| data > InsertDatabaseName.db | The SQLite3 database to save the clean data |
| models | The directory contaning the scripts to run the Text Processing, Natural Language Processing (NLP), and Machine Learning (ML) pipelines |
| models > train_classifier.py | A machine learning pipeline that loads data from a SQLite database, splits the dataset into training and test datasets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, and exports the final model as a pickle file |
| models > classifier.pkl | The saved model pickle file |
| Procfile | Instructs the runtime to use gunicorn to run our dashboard |
| README.md | The file you are currently reading |
| requirements.txt | Contains the list of required libraries as listed in the "Libraries Used" section above but includes the versions required at run time |
| runtime.txt | The Python runtime version being used |

# Summary of the results
The dataset given was imbalanced (i.e. some labels like water have few examples and others like search_and_rescue, security, child_alone, shelter, clothing, etc. had none). We discovered this when first evaluating our model and seeing Scikit warnings that read "UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples." This imbalance affected training the model because our overall precision, recall, f1-score were skewed (with so many 0 results the averages were pulled down). Unlike with other data like financials, temperature readings there really is no way to necessarily impute the data. I cannot simply average out these gaps or even do other imputing strategies like fill forward or fill back data. NLP or Natural Language Processing does not give us these imputing options. Best we might do here to get a better evaluation result would be to emphasize the stats on the categories we know are not missing by passing the labels for the categories we do have.

# Acknowledgements
Several code snippets came from previous lessons in our Udacity Data Scientist program. Also, where employed I have credited various contributors from StackOverflow.com, geeksforgeeks.org at https://www.geeksforgeeks.org/, and the Data Science Stack Exchange at https://datascience.stackexchange.com. A big thank you to our instructors and all those involved in the Udacity program.

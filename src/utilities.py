import re
import math
import unidecode
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mode
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.utils import resample
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
warnings.filterwarnings('ignore')

round_number = 3
random_state = 42
original_stopwords = stopwords.words('english')
additional_stopwords = ['none']
original_stopwords += additional_stopwords
stopwords_ = list(set(original_stopwords))

def standardize_bucket(bucket):
    if ((bucket == '1.0') | (bucket == '1')):
        return '1'
    elif ((bucket == '2') | (bucket == '3') | (bucket == '2.0') | (bucket == '3.0')):
        return '2 or 3'
    else:
        return bucket


def standardize_sent(sent):
    if ((sent == 0) | (sent == 1) | (sent == 2)):
        return 'Negative'
    elif (sent == 3):
        return 'Neutral'
    else:
        return 'Positive'


def clean_text(text):
    if type(text) == np.float64:
        return ""
    temp = text.lower() # to lower case
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp) # remove @s
    temp = re.sub("#[A-Za-z0-9_]+","", temp) # remove hashtags
    temp = re.sub(r'http\S+', '', temp) # remove links
    temp = re.sub(r"www.\S+", "", temp) # remove links
    temp = re.sub(r'\n|[^a-zA-Z]', ' ', temp) # remove punctuation
    temp = temp.replace("\n", " ").split()
    temp = [w for w in temp if not w in stopwords_] # remove stopwords
    temp = [w for w in temp if not w.isdigit()] # remove numbers
    temp = [unidecode.unidecode(w) for w in temp] # turn non-enlish letters to english letters
    temp = " ".join(word for word in temp)
    return temp


def partition_data(df, ratio, time):
    #partiton
    if time:
        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    df_rows = df.shape[0]
    seed_num = math.floor(df_rows * ratio[0])
    seed = df[:seed_num]
    unlabeled_num = seed_num + (math.floor(df_rows * ratio[1]))
    unlabeled = df[seed_num:unlabeled_num]
    test = df[unlabeled_num:]
    return seed, unlabeled, test


def calc_entropy(lst):
    unique_num = list(set(lst))
    entropy = 0
    for i in range(len(unique_num)):
        label = unique_num[i]
        prob = sum(np.array(lst) == label)/len(lst)
        entropy += prob * math.log2(1/prob)
    return entropy


def train_model(seed, task):
    if task == 1:
        target = 'Bucket'
    if task == 2:
        target = 'SentimentScore'

    cv = 5
    train, test = train_test_split(seed, random_state=random_state, test_size=0.2, shuffle=True)
    X_train, X_test, Y_train, Y_test = train[['text_cleaned']], test[['text_cleaned']], train[[target]], test[[target]]
    #Wrap in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("tf", CountVectorizer(stop_words=stopwords_), 'text_cleaned'),
            ("tfidf", TfidfVectorizer(stop_words=stopwords_), 'text_cleaned')]
    )
    #Define the model
    model_lst = [
                SVC(),
                KNeighborsClassifier(),
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                AdaBoostClassifier(),
            ]
    
    pl_preds = []
    for model in model_lst:
        #Build the pipeline
        pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('clf', OneVsRestClassifier(model, n_jobs=1)),
                ])
        #Train the model
        pipeline.fit(X_train, Y_train)
        # compute the testing accuracy
        prediction = pipeline.predict(pd.DataFrame(X_test))
        pl_preds.append([pipeline, prediction])
        
    #Saves all the model pipelines
    pipelines = [x[0] for x in pl_preds]
    #Saves all the model predictions
    all_preds = np.array([x[1] for x in pl_preds]).transpose()
    #Find the mode in all preds
    final_preds = [mode(i) for i in all_preds]
    accuracy = accuracy_score(Y_test,final_preds)
    return pipelines, accuracy


def choose_unlabeled(pipelines, unlabeled, task):
    if task == 1:
        target = 'Bucket'
    if task == 2:
        target = 'SentimentScore'   

    unlabeled_x = unlabeled[['text_cleaned']]
    unlabeled_y = unlabeled[[target]]
    all_preds = np.array([pl.predict(unlabeled_x) for pl in pipelines]).transpose()
    unlabeled['all_preds'] = list(all_preds)
    unlabeled['entropy'] = unlabeled['all_preds'].apply(calc_entropy)
    unlabeled.sort_values(by=['entropy'], ascending=False, inplace=True)


def active_learning(pipelines, seed, unlabeled, instances, task):
    if task == 1:
        target = 'Bucket'
    if task == 2:
        target = 'SentimentScore' 

    # Sort the unlabeled data based on informativeness level
    choose_unlabeled(pipelines, unlabeled, task)
    # Update the unlabeled data and the info_data
    info_data, unlabeled = unlabeled.iloc[:instances], unlabeled.iloc[instances:]
    # Add selected data to the training set
    seed = pd.concat([seed, info_data[['date', 'text', target, 'text_cleaned']]])
    pipelines, accuracy = train_model(seed)
    return pipelines, accuracy


def get_data(filepath, task):
    df_init = pd.read_csv(filepath)

    #Selects only the tweets about China
    df = df_init[df_init['country']=='China']
    df = df[['date', 'text', 'id', 'Bucket', 'SentimentScore']]

    #Shuffle the data
    df = df.sample(frac=1, replace=False, random_state=1) 
    df.reset_index(drop=True, inplace=True)

    if task == 1:
        #Standardized the bucket label
        df['Bucket'] = df['Bucket'].apply(standardize_bucket)
        #Remove tweets that are in both buckets
        df_bucket_count = pd.DataFrame(df.groupby('id')['Bucket'].nunique())
        df_bucket_count.reset_index(inplace=True)
        df_bucket_count.columns = ['tweet_id', 'bucket_num']
        df = df.merge(df_bucket_count, left_on='id', right_on='tweet_id')
        df = df[df['bucket_num'] == 1]
        #Remove tweets without a bucket (null)
        df = df[(df['Bucket'] == '1') | (df['Bucket'] == '2 or 3')]
        #Remove duplicates
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        df = df[['date', 'text', 'Bucket']]
        df["text_cleaned"] = [clean_text(t) for t in df["text"]]
    
    if task == 2:
        #Step 1: Remove tweets that do not have sentiment score
        #Step 2: Average the sentiment score for each unique tweet
        df = df.copy()[['date', 'text', 'id', 'SentimentScore']]
        df.dropna(subset=['SentimentScore'], inplace=True)

        df = pd.DataFrame(df.groupby(['date', 'text', 'id'])['SentimentScore'].mean())
        df.reset_index(inplace=True)

        #Remove ambiguous labels
        range_lst = [0, 1, 2, 3, 4, 5]
        df = df[df['SentimentScore'].apply(lambda x: True if x in range_lst else False)]
        df['SentimentScore'] = df['SentimentScore'].apply(standardize_sent)

        #Remove duplicates
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        df = df[['date', 'text', 'SentimentScore']]
        df["text_cleaned"] = [clean_text(t) for t in df["text"]]

    return df
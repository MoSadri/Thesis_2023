
"""
This file contains code to run the cross validation (cv) procedure. 
The following information will be printed out for each fold of the cv procedure
1. Accuracy
2. Precision
3. Recall
4. F1 Score
"""

from calendar import c
import sys
import os
from pathlib import Path
print("Python version")
print(sys.version)
print("Version info:")
print(sys.version_info)
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer
import string
import re
sys.setrecursionlimit(1500)
import pickle
import matplotlib.pyplot as plt
import seaborn

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Ensure k is the number of fold you want to do, make sure this is the same number as used in generate_cv_data.py.
k = 5
# Typically I use data_name to signify which data I used, like 'black' means the file black_dataset.csv, 'balanced' means the file balanced_dataset.csv
#data_name = 'black'
#data_name = 'women'
data_name = 'balanced'

# Directory for saving the output
cv_output_path = '../cv_output'

stopwords=stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

sentiment_analyzer = VS()
stemmer = PorterStemmer()

# Confusion matrix file name, should no longer need to use this
confusion_file = 'confusion_matrix.pdf'

# def preprocess(text_string):
#     """
#     Accepts a text string and replaces:
#     1) urls with URLHERE
#     2) lots of whitespace with one instance
#     3) mentions with MENTIONHERE
#
#     This allows us to get standardized counts of urls and mentions
#     Without caring about specific people mentioned
#     """
#     space_pattern = '\s+'
#     giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
#         '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#     mention_regex = '@[\w\-]+'
#     parsed_text = re.sub(space_pattern, ' ', text_string)
#     parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
#     parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
#     #parsed_text = parsed_text.code("utf-8", errors='ignore')
#     return parsed_text.decode('utf-8', 'ignore')

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    mention_regex = r'@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    # parsed_text = parsed_text.encode("utf-8", errors='ignore').decode('utf-8', 'ignore')
    return parsed_text




# def tokenize(tweet):
#     """Removes punctuation & excess whitespace, sets to lowercase,
#     and stems tweets. Returns a list of stemmed tokens."""
#     tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
#     #tokens = re.split("[^a-zA-Z]*", tweet.lower())
#     tokens = [stemmer.stem(t) for t in tweet.split()]
#     return tokens

import re

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens



def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]+", tweet.lower())).strip()
    return tweet.split()

# def get_pos_tags(tweets):
#     """Takes a list of strings (tweets) and
#     returns a list of strings of (POS tags).
#     """
#     tweet_tags = []
#     for t in tweets:
#         tokens = basic_tokenize(preprocess(t))
#         tags = nltk.pos_tag(tokens)
#         tag_list = [x[1] for x in tags]
#         #for i in range(0, len(tokens)):
#         tag_str = " ".join(tag_list)
#         tweet_tags.append(tag_str)
#     return tweet_tags

import nltk
from nltk.tokenize import word_tokenize

def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = word_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags



def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features_(tweet, oth_features):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features.

    This is modified to only include those features in the final
    model."""

    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet) #Get text only

    syllables = textstat.syllable_count(words) #count syllables in words
    num_chars = sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://

    retweet = 0
    if "rt" in words:
        retweet = 1

    features = []

    if "FKRA" in oth_features:
        features.append(FKRA)
    if "FRE" in oth_features:
        features.append(FRE)
    if "num_syllables" in oth_features:
        features.append(syllables)
    if "avg_syl_per_word" in oth_features:
        features.append(avg_syl)
    if "num_chars" in oth_features:
        features.append(num_chars)
    if "num_chars_total" in oth_features:
        features.append(num_chars_total)
    if "num_terms" in oth_features:
        features.append(num_terms)
    if "num_words" in oth_features:
        features.append(num_words)
    if "num_unique_words" in oth_features:
        features.append(num_unique_terms)
    if "vader neg" in oth_features:
        features.append(sentiment['neg'])
    if "vader pos" in oth_features:
        features.append(sentiment['pos'])
    if "vader neu" in oth_features:
        features.append(sentiment['neu'])
    if "vader compound" in oth_features:
        features.append(sentiment['compound'])
    if "num_hashtags" in oth_features:
        features.append(twitter_objs[2])
    if "num_mentions" in oth_features:
        features.append(twitter_objs[1])
    if "num_urls" in oth_features:
        features.append(twitter_objs[0])
    if "is_retweet" in oth_features:
        features.append(retweet)

    return features

def get_oth_features(tweets, oth_features):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats=[]
    for t in tweets:
        feats.append(other_features_(t, oth_features))
    return np.array(feats)


def transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer):
    """
    This function takes a list of tweets, along with used to
    transform the tweets into the format accepted by the model.

    Each tweet is decomposed into
    (a) An array of TF-IDF scores for a set of n-grams in the tweet.
    (b) An array of POS tag sequences in the tweet.
    (c) An array of features including sentiment, vocab, and readability.

    Returns a pandas dataframe where each row is the set of features
    for a tweet. The features are a subset selected using a Logistic
    Regression with L1-regularization on the training data.

    """
    tf_array = tf_vectorizer.fit_transform(tweets).toarray()
    tfidf_array = tf_array*idf_vector
    print ("Built TF-IDF array")

    pos_tags = get_pos_tags(tweets)
    pos_array = pos_vectorizer.fit_transform(pos_tags).toarray()
    print ("Built POS array")
    
    oth_features = joblib.load(oth_pkl_file)
    oth_array = get_oth_features(tweets, oth_features)
    print ("Built other feature array")
   
    M = np.concatenate([tfidf_array, pos_array, oth_array],axis=1)
    return pd.DataFrame(M)

def predictions(X, model):
    """
    This function calls the predict function on
    the trained model to generated a predicted y
    value for each observation.
    """

    y_preds = model.predict(X)
    return y_preds

def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Hate"
    elif class_label == 1:
        return "Offensive"
    elif class_label == 2:
        return "Neither"
    else:
        return "NA"

# def get_tweets_predictions(tweets, perform_prints=True):
#     fixed_tweets = []
#     for i, t_orig in enumerate(tweets):
#         s = t_orig
#         try:
#             s = s.encode("latin1")
#         except:
#             try:
#                 s = s.encode("utf-8")
#             except:
#                 pass
#         if type(s) != str:
#             fixed_tweets.append(str(s, errors="ignore"))
#         else:
#             fixed_tweets.append(s)
#     assert len(tweets) == len(fixed_tweets), "shouldn't remove any tweets"
#     tweets = fixed_tweets
#     print (len(tweets), " tweets to classify")

def get_tweets_predictions(tweets, perform_prints=True):
    fixed_tweets = []
    for i, t_orig in enumerate(tweets):
        if isinstance(t_orig, str):
            fixed_tweets.append(t_orig)
        else:
            fixed_tweets.append(t_orig.decode('utf-8', 'ignore'))
    assert len(tweets) == len(fixed_tweets), "shouldn't remove any tweets"
    tweets = fixed_tweets
    print (len(tweets), " tweets to classify")


    print ("Loading trained classifier... ")
    model = joblib.load(model_pkl_file)
    print ("Loading other information...")
    tf_vectorizer = joblib.load(tfidf_pkl_file)
    idf_vector = joblib.load(idf_pkl_file)
    pos_vectorizer = joblib.load(pos_pkl_file)
    #Load ngram dict
    #Load pos dictionary
    #Load function to transform data

    print ("Transforming inputs...")
    X = transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer)

    print ("Running classification model...")
    predicted_class = predictions(X, model)

    return predicted_class

def create_confusion_matrix(y, y_preds):
    plt.rc('pdf', fonttype=42)
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.serif'] = 'Times'
    plt.rcParams['font.family'] = 'serif'
    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(y,y_preds)
    matrix_proportions = np.zeros((3,3))
    for i in range(0,3):
        matrix_proportions[i,:] = confusion_matrix[i,:]/float(confusion_matrix[i,:].sum())
    names=['Hate','Offensive','Neither']
    confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
    plt.figure(figsize=(5,5))
    seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
    plt.ylabel(r'\textbf{True categories}',fontsize=14)
    plt.xlabel(r'\textbf{Predicted categories}',fontsize=14)
    plt.tick_params(labelsize=12)    
    plt.savefig(confusion_file)

def process_tweets(input_file, i):
    df = pd.read_csv(input_file, encoding='latin1')
    
    if hasattr(df, 'text'):
        tweets = df.text
    else:
        tweets = df.tweet
    tweets = [x for x in tweets if type(x) == str]
    predictions = get_tweets_predictions(tweets)
    real_class = df['class'].values

    print ("Saving predicted values: ")
    # Shouldn't need to print out every tweet
    #for i,t in enumerate(trump_tweets):
        #print (t)
        #print (class_to_name(trump_predictions[i]))

    classifications = {'Hate': 0, 'Offensive': 0, 'Neither': 0, 'NA': 0}
    for i, t in enumerate(tweets):
        classifications[class_to_name(predictions[i])] += 1

    p = Path(input_file)

    #output_file = f'fold{fold_num}results.txt'

    accuracy = accuracy_score(real_class, predictions)
    precision = precision_score(real_class, predictions, pos_label=0)
    recall = recall_score(real_class, predictions, pos_label=0)
    f1 = f1_score(real_class, predictions, pos_label=0)

    with open(output_file, "w") as f:
        total_tweets = sum(classifications.values())
        f.write(f"Accuracy = {accuracy}\n")
        f.write(f"Precision = {precision}\n")
        f.write(f"Recall = {recall}\n")
        f.write(f"F1 Score = {f1}")

def process_labeled_data():
    print ("Calculate accuracy on labeled data")
    df = pd.read_csv('labeled_data.csv')
    tweets = df['tweet'].values
    tweets = [x for x in tweets if type(x) == str]
    tweets_class = df['class'].values
    print(tweets_class)
    predictions = get_tweets_predictions(tweets)
    right_count = 0

    for i,t in enumerate(tweets):
        if tweets_class[i] == predictions[i]:
            right_count += 1

    accuracy = right_count / float(len(df))
    print ("accuracy", accuracy)

    #create_confusion_matrix(tweets_class, predictions)

if __name__ == '__main__':
    for i in range(1, k+1):

        fold_num = i
        # Pickled file names
        model_pkl_file = f'../cv_data/{data_name}_cvtrain_fold{i}_model.pkl'
        tfidf_pkl_file = f'../cv_data/{data_name}_cvtrain_fold{i}_tfidf.pkl'
        idf_pkl_file = f'../cv_data/{data_name}_cvtrain_fold{i}_idf.pkl'
        pos_pkl_file = f'../cv_data/{data_name}_cvtrain_fold{i}_pos.pkl'
        oth_pkl_file = f'../cv_data/{data_name}_cvtrain_fold{i}_oth.pkl'

        input_file = f'../cv_data/{data_name}_cvanalysis_fold{i}.csv'

        # Create the output directory if it doesn't exist
        if not os.path.exists(cv_output_path):
            # If not, create the directory
            os.makedirs(cv_output_path)

        output_file = f'{cv_output_path}/{data_name}_cvresults_fold{i}.txt'

        print('Processing ' + input_file)
        process_tweets(input_file, i)

    print("Finish doing cross validation")
    #process_labeled_data()
    

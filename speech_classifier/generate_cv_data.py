"""
This file contains code to generate the data need for performing cross validation (cv).
This requires an input file in the format of data_name_dataset.csv, and will produce k number of training sets and analysis sets.
For each training set, 5 .pkl files will be produced, these .pkl files are the trained data, 
and will be used in the code running the actual cv process..
"""

from itertools import takewhile
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# Number of fold validation desired
k = 5
# Typically I use data_name to signify which data I used, like 'black' means the file black_dataset.csv, 'balanced' means the file balanced_dataset.csv
#data_name = 'black'
data_name = 'women'
#data_name = 'balanced'

data_file = f'../data/{data_name}_dataset.csv'


stopwords=stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()

sentiment_analyzer = VS()

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
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]+", tweet.lower())).strip()
    return tweet.split()

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

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    ##SENTIMENT
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
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

def generate_pkl_files(train_data, fold):
    tweets = train_data.text

    model_file = f'../cv_data/{data_name}_cvtrain_fold{fold+1}_model.pkl'
    tfidf_file = f'../cv_data/{data_name}_cvtrain_fold{fold+1}_tfidf.pkl'
    idf_file = f'../cv_data/{data_name}_cvtrain_fold{fold+1}_idf.pkl'
    pos_file = f'../cv_data/{data_name}_cvtrain_fold{fold+1}_pos.pkl'
    oth_file = f'../cv_data/{data_name}_cvtrain_fold{fold+1}_oth.pkl'

    vectorizer = TfidfVectorizer(
        #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords, #We do better when we keep stopwords
        use_idf=True,
        smooth_idf=False,
        norm=None, #Applies l2 norm smoothing
        decode_error='replace',
        max_features=10000,
        min_df=5,
        max_df=0.501
        )

    #Construct tfidf matrix and get relevant scores
    tfidf = vectorizer.fit_transform(tweets).toarray()
    vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names_out())}
    idf_vals = vectorizer.idf_
    idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores

    #Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
            #print(tokens[i],tag_list[i])

    #We can use the TFIDF vectorizer to get a token matrix for the POS tags
    pos_vectorizer = TfidfVectorizer(
        #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None, #We do better when we keep stopwords
        use_idf=False,
        smooth_idf=False,
        norm=None, #Applies l2 norm smoothing
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.501,
        )

    #Construct POS TF matrix and get vocab dict
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names_out())}

    other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                            "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", "vader compound", \
                            "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

    feats = get_feature_array(tweets)
    pos_full = pos_vectorizer.get_feature_names_out()

    #Now join them all up
    M = np.concatenate([tfidf,pos,feats],axis=1)

    #Finally get a list of variable names
    variables = ['']*len(vocab)
    for k,v in vocab.items():
        variables[v] = k

    pos_variables = ['']*len(pos_vocab)
    for k,v in pos_vocab.items():
        pos_variables[v] = k

    feature_names = variables+pos_variables+other_features_names

    X = pd.DataFrame(M)
    y = train_data['class'].astype(int)

    select = SelectFromModel(LogisticRegression(class_weight='balanced',solver='liblinear',penalty="l1",C=0.01))
    X_ = select.fit_transform(X,y)

    model = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr').fit(X_, y)
    joblib.dump(model, model_file)
    y_preds = model.predict(X_)
    report = classification_report( y, y_preds )
    final_features = select.get_support(indices=True) #get indices of features
    final_feature_list = [(feature_names[i]) for i in final_features] #Get list of names corresponding to indices

    #Getting names for each class of features
    pos_indices = []
    for item in pos_full:
        if item in final_feature_list:
            # If item is found, append its lowest index to the indices list
            pos_indices.append(final_feature_list.index(item))

    pos_min_index = min(pos_indices)
    pos_max_index = max(pos_indices)
    ngram_features = final_feature_list[:pos_min_index]
    pos_features = final_feature_list[pos_min_index:pos_max_index+1]
    oth_features = final_feature_list[pos_max_index+1:]

    joblib.dump(oth_features, oth_file) 

    new_vocab = {v:i for i, v in enumerate(ngram_features)}
    new_vocab_to_index = {}
    for k in ngram_features:
        new_vocab_to_index[k] = vocab[k]

    #Get indices of text features
    ngram_indices = final_features[:len(ngram_features)]

    new_vectorizer = TfidfVectorizer(
        #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords, #We do better when we keep stopwords
        use_idf=False,
        smooth_idf=False,
        norm=None, #Applies l2 norm smoothing
        decode_error='replace',
        min_df=1,
        max_df=1.0,
        vocabulary=new_vocab
        )

    joblib.dump(new_vectorizer, tfidf_file) 
    idf_vals_ = idf_vals[ngram_indices]
    joblib.dump(idf_vals_, idf_file) 

    # ## Generating POS features
    # This is simpler as we do not need to worry about IDF but it will be slower as we have to compute the POS tags for the new data. Here we can simply use the old POS tags.
    new_pos = {v:i for i, v in enumerate(pos_features)}
    #We can use the TFIDF vectorizer to get a token matrix for the POS tags
    new_pos_vectorizer = TfidfVectorizer(
        #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None, #We do better when we keep stopwords
        use_idf=False,
        smooth_idf=False,
        norm=None, #Applies l2 norm smoothing
        decode_error='replace',
        min_df=1,
        max_df=1.0,
        vocabulary=new_pos
        )

    joblib.dump(new_pos_vectorizer, pos_file) 



if __name__ == '__main__':

    # Read the CSV file
    data = pd.read_csv(data_file)

    # Shuffle the data randomly
    data = data.sample(frac=1).reset_index(drop=True)

    # Initialize the k-fold cross-validation
    kfold = KFold(n_splits=k)

    # Iterate over the splits
    for fold, (train_indices, test_indices) in enumerate(kfold.split(data)):
        # Create train and test dataframes for the current fold
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

        # Define the directory path
        cvdata_path = "../cv_data"

        # Check if the directory exists
        if not os.path.exists(cvdata_path):
            # If not, create the directory
            os.makedirs(cvdata_path)

        # Technically this cvtrain file doesn't need to be outputted, but it can be useful someone wants to check its content
        train_data.to_csv(f'{cvdata_path}/{data_name}_cvtrain_fold{fold+1}.csv', index=False)

        # Test data is needed for the run_cv.py
        test_data.to_csv(f'{cvdata_path}/{data_name}_cvanalysis_fold{fold+1}.csv', index=False)

        # Now generate the .pkl files needed
        generate_pkl_files(train_data, fold)


    print("Finish")
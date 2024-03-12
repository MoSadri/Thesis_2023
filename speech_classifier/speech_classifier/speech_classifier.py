
#%%

#%%
"""
This file contains code to

    (a) Load the pre-trained classifier and
    associated files.

    (b) Transform new input data into the
    correct format for the classifier.

    (c) Run the classifier on the transformed
    data and return results.
"""


# `import numpy as np
# import pandas as pd
# import joblib
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# from nltk.stem import PorterStemmer
# import string
# import re
# import os
# import sys
# sys.setrecursionlimit(1500)
# import _pickle as pickle
#
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
# import _pickle as pickle
# from textstat import *

from calendar import c
import sys
import os
from pathlib import Path
print(sys.version)
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

stopwords=stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

sentiment_analyzer = VS()

stemmer = PorterStemmer()

# Typically I use data_name to signify which data I used, like 'black' means the file black_dataset.csv, 'balanced' means the file balanced_dataset.csv
# Default to balanced if no argument is given
data_name = 'balanced'
if len(sys.argv) > 1:
    data_name = sys.argv[1]

# Pickled file names
model_pkl_file = f'../data/{data_name}_model.pkl'
tfidf_pkl_file = f'../data/{data_name}_tfidf.pkl'
idf_pkl_file = f'../data/{data_name}_idf.pkl'
pos_pkl_file = f'../data/{data_name}_pos.pkl'
oth_pkl_file = f'../data/{data_name}_oth.pkl'

# Directory names
input_directory = '../input'
output_directory = f'../output/results_{data_name}'

# Confusion matrix file name
#confusion_file_name = f'confusion_{data_name}.pdf'

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
    print("tf_array size: ", tf_array.shape)

    pos_tags = get_pos_tags(tweets)
    pos_array = pos_vectorizer.fit_transform(pos_tags).toarray()
    print("pos_array size: ", pos_array.shape)
    print ("Built POS array")
    
    oth_features = joblib.load(oth_pkl_file)
    oth_array = get_oth_features(tweets, oth_features)
   
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

def create_confusion_matrix(y, y_preds, nclass, output_file):
    plt.rc('pdf', fonttype=42)
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.serif'] = 'Times'
    plt.rcParams['font.family'] = 'serif'
    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(y,y_preds)
    matrix_proportions = np.zeros((nclass, nclass))
    for i in range(0,nclass):
        matrix_proportions[i,:] = confusion_matrix[i,:]/float(confusion_matrix[i,:].sum())
    
    if nclass == 3:
        names=['Hate','Offensive','Neither']
    else:
        names=['Hate+Offensive', 'Neither']
    confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
    plt.figure(figsize=(5,5))
    seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
    plt.ylabel(r'\textbf{True categories}',fontsize=14)
    plt.xlabel(r'\textbf{Predicted categories}',fontsize=14)
    plt.tick_params(labelsize=12)    
    plt.savefig(output_file)

"""
This function assume that we are processing some tweets/text without knowing their actual class.
So we only output their predicted values here.
"""
def process_tweets(input_file):
    df = pd.read_csv(input_file, encoding='latin1')
    
    if hasattr(df, 'Text'):
        tweets = df.Text
    else:
        tweets = df.tweet
    tweets = [x for x in tweets if type(x) == str]
    predictions = get_tweets_predictions(tweets)

    # Shouldn't need to print out every tweet
    #for i,t in enumerate(trump_tweets):
        #print (t)
        #print (class_to_name(trump_predictions[i]))

    classifications = {'Hate': 0, 'Offensive': 0, 'Neither': 0, 'NA': 0}
    for i, t in enumerate(tweets):
        classifications[class_to_name(predictions[i])] += 1

    # Save the classification results to file
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    p = Path(input_file)

    output_file = output_directory + '/' + p.stem + '.txt'

    with open(output_file, "w") as f:
        total_tweets = sum(classifications.values())
        f.write(f"Total tweets = {total_tweets}\n")
        f.write(f"Hate = {classifications['Hate']}\n")
        f.write(f"Offensive = {classifications['Offensive']}\n")
        f.write(f"Neither = {classifications['Neither']}\n")
        f.write(f"Not Applicable = {classifications['NA']}")

def process_labeled_data(filename, original_3_classes=False, new_2_classes=True):
    print(f"Create confusion matrix from {input_directory}/{filename}")
    df = pd.read_csv(f'{input_directory}/{filename}')
    tweets = df['tweet'].values
    tweets = [x for x in tweets if type(x) == str]
    real_class = df['class'].values
    predictions = get_tweets_predictions(tweets)
    #right_count = 0
    
    if original_3_classes:
        confusion_output_file = output_directory + f'/scenario_{data_name}_data_{filename}_(3_classes).pdf'
        create_confusion_matrix(real_class, predictions, 3, confusion_output_file)

        accuracy = accuracy_score(real_class, predictions)
        precision = precision_score(real_class, predictions, average=None)
        recall = recall_score(real_class, predictions, average=None)
        f1 = f1_score(real_class, predictions, average=None)

        output_file = output_directory + f'/scenario_{data_name}_data_{filename}_(3_classes).txt'
        with open(output_file, "w") as f:
            f.write(f"Accuracy = {accuracy}\n")
            f.write(f"Precision = {precision}\n")
            f.write(f"Recall = {recall}\n")
            f.write(f"F1 Score = {f1}")

    if new_2_classes:
    # Replace all 1 (offensive class) with 0 (hate class), then plot the confusion matrix
        real_class[real_class == 1] = 0

        confusion_output_file = output_directory + f'/scenario_{data_name}_data_{filename}_(2_classes).pdf'
        create_confusion_matrix(real_class, predictions, 2, confusion_output_file)

        accuracy = accuracy_score(real_class, predictions)
        precision = precision_score(real_class, predictions, pos_label=0)
        recall = recall_score(real_class, predictions, pos_label=0)
        f1 = f1_score(real_class, predictions, pos_label=0)

        output_file = output_directory + f'/scenario_{data_name}_data_{filename}_(2_classes).txt'
        with open(output_file, "w") as f:
            f.write(f"Accuracy = {accuracy}\n")
            f.write(f"Precision = {precision}\n")
            f.write(f"Recall = {recall}\n")
            f.write(f"F1 Score = {f1}")


if __name__ == '__main__':
    # Trump tweets obtained here: https://github.com/sashaperigo/Trump-Tweets
        
    # Obtain input files from input directory
    files = os.listdir(input_directory)
    # Process all csv files in the input directory
    for file in files:
        print('Processing ' + input_directory + '/' + file)
        process_tweets(input_directory + '/' + file)

    # This is running through a manually labeled data for producing confusion matrix and calculate accuracies
    # Below is only supposed to be run with only pre-labeled data
    # For original trained model, it uses 3 classes and therefore cannot be run with the 2 classes implementation
    if data_name == 'original':
        process_labeled_data('labeled_data.csv', original_3_classes=True, new_2_classes=False)
        process_labeled_data('black.csv', original_3_classes=True, new_2_classes=False)
        process_labeled_data('women.csv', original_3_classes=True, new_2_classes=False)
        process_labeled_data('lgbt.csv', original_3_classes=True, new_2_classes=False)
    else:
        process_labeled_data('labeled_data.csv', original_3_classes=True)
        process_labeled_data('black.csv')
        process_labeled_data('women.csv')
        process_labeled_data('lgbt.csv')


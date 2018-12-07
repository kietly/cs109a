---
title: Models
notebook: twitter-model.ipynb
nav_include: 2
---


## Data Cleaning for NLP


* Summary
* Loading Data 
* Initial Inspection
    * Discarding Features
* Data cleaning for NLP    
* EDA/Feature Engineering
    * Sentiment Analysis

### Summary

In this section we perform NLP related data cleaning and EDA on the human and bot tweet-level dataset. Results of analyzing features indicated that the most useful fields for our classification efforts are: **truncated, retweet count, favorite count, num hashtags, num urls, and num mentions**. Engineered fields that could support classification are sentiment based fields (**ratio_positive, ratio_negative, ratio_neutral**) and **place_binary**.

### Loading Data

This dataset is comprised of about 50,000 human tweets and 60,000 bot tweets. There are 26 features and one response. The tweet content is in a column called 'text' within the tweets.csv files.  The 'user_type' is our response variable and contains a 1 if the account is a bot.

To conduct our analysis, first we import all the modules and functions we will use in this stage of analysis.



```python
import pandas as pd
import numpy as np
import langdetect
#either pip install langdetect or conda install -c conda-forge langdetect
from langdetect import detect
from pandas.plotting import scatter_matrix
import re
#conda install -c conda-forge textblob 
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import html
from html.parser import HTMLParser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import textblob
from textblob import TextBlob
import string
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
```


Next we load the data and concatenate the human and bot tweet files into a single dataframe called tweets. 



```python
#read in all the tweets from geniuine accounts
human_tweets = pd.read_csv('~/Documents/GitHub/cs109a/data/human_tweets_100.csv')
bot_tweets = pd.read_csv('~/Documents/GitHub/cs109a/data/bot_tweets_100.csv')
sbot_tweets = pd.read_csv('~/Documents/GitHub/cs109a/data/social_tweets_100.csv')
human_tweets = human_tweets.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
bot_tweets = bot_tweets.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
sbot_tweets = sbot_tweets.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
#We will combine these two data sets then get a sample to analyze
tweets_some = pd.concat([human_tweets, bot_tweets], sort=False)
tweets = pd.concat([tweets_some, sbot_tweets], sort=False)
#tweets_train, tweets_test= train_test_split(tweets, test_size=0.3, stratify=tweets['user_type'], random_state=5)
```


### Initial Inspection
Using the .info method, we can do a quick analysis of the columns and see some predictors we can already eliminate.



```python
tweets.info()
```


    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 120394 entries, 0 to 10041
    Data columns (total 27 columns):
    id                         120394 non-null int64
    text                       120260 non-null object
    source                     120394 non-null object
    user_id                    120394 non-null float64
    truncated                  354 non-null float64
    in_reply_to_status_id      120394 non-null int64
    in_reply_to_user_id        120394 non-null int64
    in_reply_to_screen_name    19419 non-null object
    retweeted_status_id        120394 non-null int64
    geo                        0 non-null float64
    place                      1920 non-null object
    contributors               0 non-null float64
    retweet_count              120394 non-null int64
    reply_count                120394 non-null int64
    favorite_count             120394 non-null int64
    favorited                  0 non-null float64
    retweeted                  0 non-null float64
    possibly_sensitive         0 non-null float64
    num_hashtags               120394 non-null int64
    num_urls                   120394 non-null int64
    num_mentions               120394 non-null int64
    created_at                 120394 non-null object
    timestamp                  120394 non-null object
    crawled_at                 120394 non-null object
    updated                    120394 non-null object
    user_type                  120394 non-null int64
    counts                     60944 non-null float64
    dtypes: float64(8), int64(11), object(8)
    memory usage: 25.7+ MB




```python
tweets.describe()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>truncated</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_user_id</th>
      <th>retweeted_status_id</th>
      <th>geo</th>
      <th>contributors</th>
      <th>retweet_count</th>
      <th>reply_count</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>retweeted</th>
      <th>possibly_sensitive</th>
      <th>num_hashtags</th>
      <th>num_urls</th>
      <th>num_mentions</th>
      <th>user_type</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.203940e+05</td>
      <td>1.203940e+05</td>
      <td>354.0</td>
      <td>1.203940e+05</td>
      <td>1.203940e+05</td>
      <td>1.203940e+05</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.203940e+05</td>
      <td>120394.0</td>
      <td>120394.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120394.000000</td>
      <td>120394.000000</td>
      <td>120394.000000</td>
      <td>120394.000000</td>
      <td>60944.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.192202e+17</td>
      <td>6.175252e+08</td>
      <td>1.0</td>
      <td>5.348284e+16</td>
      <td>8.755713e+07</td>
      <td>4.592845e+16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.223074e+02</td>
      <td>0.0</td>
      <td>0.116235</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.202867</td>
      <td>0.491362</td>
      <td>0.460214</td>
      <td>0.410386</td>
      <td>1900.994224</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.432904e+17</td>
      <td>9.033170e+08</td>
      <td>0.0</td>
      <td>1.519032e+17</td>
      <td>3.679532e+08</td>
      <td>1.449689e+17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.771989e+04</td>
      <td>0.0</td>
      <td>0.732170</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.635952</td>
      <td>0.517175</td>
      <td>0.874003</td>
      <td>0.491906</td>
      <td>1229.092513</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.460111e+06</td>
      <td>8.872810e+05</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.076543e+09</td>
      <td>3.896226e+07</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>669.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.785777e+10</td>
      <td>7.895294e+07</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2221.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.782912e+17</td>
      <td>6.044007e+08</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3200.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.941273e+17</td>
      <td>2.386078e+09</td>
      <td>1.0</td>
      <td>5.940862e+17</td>
      <td>3.198892e+09</td>
      <td>5.941085e+17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.350110e+06</td>
      <td>0.0</td>
      <td>54.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.000000</td>
      <td>4.000000</td>
      <td>17.000000</td>
      <td>1.000000</td>
      <td>3200.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Features to discard for this analysis

Analyzing the .info() output shows us that five of the features are empty or have 0 for every value: 'geo', 'contributors', 'favorited', 'possibly_sensitive', 'retweeted', 'reply_count'.

Since we are not performing any network analysis, we will also drop the 'in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id', which are fields that can be used to reconstruct communications connectivity of an individual tweet. 

Other fields that are researcher or twitter generated metadata that we will remove are id:(tweet id), source:(unknown), created_at, crawled_at, updated. Since we are not performing time series analysis, we will also remove timestamp. 



```python
remove = ['geo', 'contributors', 'favorited', 'counts', 'possibly_sensitive', 'retweeted', 'reply_count','in_reply_to_screen_name', 'in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id', 'id', 'source', 'created_at', 'crawled_at', 'updated', 'timestamp']
tweets = tweets.drop(columns=remove, axis=1)
```


### Data Cleaning for NLP

Next, we can start the cleaning process on the content of the tweets themselves. This is a necessary processing step to prepare the data for follow-on NLP techniques. We've written some functions, below, that will strip punctuation, url encoding, urls, stopwords, and lemmatize our data. 



```python
#Functions
class MLStripper(HTMLParser):
    #https://docs.python.org/3/library/html.parser.html
    #https://stackoverflow.com/questions/11061058/using-htmlparser-in-python-3-2
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
       
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def remove_swords(text):
    #https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    stop_words = set(stopwords.words('english'))   
    word_tokens = word_tokenize(text)  
    filtered_sentence = [w for w in word_tokens if not w in stop_words]   
    return filtered_sentence

def get_tweet_sentiment(tweet): 
        ''' 
        https://medium.freecodecamp.org/basic-data-analysis-on-twitter-with-python-251c2a85062e 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(tweet) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'

def lemmatize(text):
    #https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
    text_out = []
    def get_lemma(word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma
    
    for word in text:
        lword = get_lemma(word)
        text_out.append(lword)
    return text_out

def nlp_clean(text):
    #get rid of #retweets RT @[\S]+  mentions @[\S]+ urls http:\S+|https\S+|www.\S+ punctuation
    text_out = []
    result = ''
    try:
        result = re.sub(r"RT @[\S]+: |@[\S]+|http:\S+|https\S+|www.\S+|[^\w\s]", "", text) 
        
         #to lower case
        result = result.lower()
        
        #get rid of url encoding
        #https://stackoverflow.com/questions/11061058/using-htmlparser-in-python-3-2
        result = strip_tags(result)

        #get rid of special ascii characters
        result = ''.join([c for c in result if ord(c) < 128])

       
        #get rid of stopwords
        result = remove_swords(result)

        #get word roots
        result = lemmatize(result)
        
    except:
        text_out = ['Failed']
    
    return result

def clean_tweets(df, text_col):
    #creates two new features
    #word_bag is our bag of words that has been cleaned
    #sentiment is the sentiment for the individual tweet
    df['word_bag'] = df[text_col].apply(lambda x: nlp_clean(x))
    
    return(df) 

```


Now we apply our NLP pre-processing



```python
tweets = clean_tweets(tweets, 'text')
```


### EDA/Feature Engineering


#### Sentiment Analysis
Now we conduct sentiment analysis using the built-in sentiment method from the textblob module, an nltk wrapper. This method predicts sentiment based on a model trained on a labeled movie review corpus. First we load our functions.



```python
#get sentiment
def sentiment(text):
    sentiment = get_tweet_sentiment(text)
    return sentiment 

#Functions to compute summary sentiment percentages  
#per user, per dataframe
def sum_sent(row):
    return row['neg_count'] + row['pos_count'] + row['ntl_count']

def ratio_pos(row):
    return row['pos_count']/row['sentiment_sum']

def ratio_neg(row):
    return row['neg_count']/row['sentiment_sum']

def ratio_neu(row):
    return row['ntl_count']/row['sentiment_sum']

def compute_sentiment_percentages(df, text_col, sentiment_col, user_id_col):
    #measure sentiment, then create dummy variables    
    df['sentiment'] = df[text_col].astype(str).apply(lambda x: sentiment(x))
    df = pd.get_dummies(df, columns=[sentiment_col])
    #get counts for each sentiment
    df['neg_count']=df.groupby([user_id_col])['sentiment_negative'].transform('sum')
    df['pos_count']=df.groupby([user_id_col])['sentiment_positive'].transform('sum')
    df['ntl_count']=df.groupby([user_id_col])['sentiment_neutral'].transform('sum')
    #compute sum
    df['sentiment_sum'] = df.apply(sum_sent, axis=1)
    df['ratio_pos'] = df.apply(ratio_pos, axis=1)
    df['ratio_neg'] = df.apply(ratio_neg, axis=1)
    df['ratio_neu'] = df.apply(ratio_neu, axis=1)
    return df
```


Then apply the sentiment analysis and compute overall percentages per user.



```python
tweets = compute_sentiment_percentages(tweets, 'text', 'sentiment', 'user_id')
```




```python
tweets.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>user_id</th>
      <th>truncated</th>
      <th>place</th>
      <th>retweet_count</th>
      <th>favorite_count</th>
      <th>num_hashtags</th>
      <th>num_urls</th>
      <th>num_mentions</th>
      <th>user_type</th>
      <th>...</th>
      <th>sentiment_negative</th>
      <th>sentiment_neutral</th>
      <th>sentiment_positive</th>
      <th>neg_count</th>
      <th>pos_count</th>
      <th>ntl_count</th>
      <th>sentiment_sum</th>
      <th>ratio_pos</th>
      <th>ratio_neg</th>
      <th>ratio_neu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>How Randolph Hodgson and Neals Yard Dairy gave...</td>
      <td>887281.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>87.0</td>
      <td>216.0</td>
      <td>323.0</td>
      <td>626.0</td>
      <td>0.345048</td>
      <td>0.138978</td>
      <td>0.515974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>“Twitter’s multi-billion dollar mistake happen...</td>
      <td>887281.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>87.0</td>
      <td>216.0</td>
      <td>323.0</td>
      <td>626.0</td>
      <td>0.345048</td>
      <td>0.138978</td>
      <td>0.515974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The evolution of advertising in the legal sect...</td>
      <td>887281.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>87.0</td>
      <td>216.0</td>
      <td>323.0</td>
      <td>626.0</td>
      <td>0.345048</td>
      <td>0.138978</td>
      <td>0.515974</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RT @rorysutherland: Plan Bee - http://t.co/030...</td>
      <td>887281.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>87.0</td>
      <td>216.0</td>
      <td>323.0</td>
      <td>626.0</td>
      <td>0.345048</td>
      <td>0.138978</td>
      <td>0.515974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RT @davewiner: Some say the Other Internet is ...</td>
      <td>887281.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>87.0</td>
      <td>216.0</td>
      <td>323.0</td>
      <td>626.0</td>
      <td>0.345048</td>
      <td>0.138978</td>
      <td>0.515974</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Since the sentiment percentages are computed from the counts, let's drop the sentiment counts before we analyze correlation.



```python
tweets = tweets.drop(['neg_count','pos_count','ntl_count','sentiment_sum'], axis=1)
```


Also, let's convert the 'place' feature into a boolean.



```python
tweets['place_bin'] = tweets.place.apply(lambda x: 1 if isinstance(x, str) else 0)
```


Now we can analyze the features. First, let's generate a scatter matrix from the non-textual or id based fields. 



```python
tweets_corr = tweets.drop(['text','user_id','place','word_bag'], axis=1)
tweets_scatter = tweets_corr.drop(['ratio_pos','ratio_neg','ratio_neu'], axis=1)
#scatter_matrix(tweets_scatter)
```




```python
tweets_bots = tweets_corr[tweets_corr['user_type']==0]
tweets_bots = tweets_bots.drop(['user_type'], axis=1)
plt.matshow(tweets_bots.corr())
plt.title('Correlation Matrix for Tweet Bots', y=1.5)
plt.xticks(range(len(tweets_bots.columns)), tweets_bots.columns);
plt.yticks(range(len(tweets_bots.columns)), tweets_bots.columns);
plt.xticks(rotation=90)
```





    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
     <a list of 13 Text xticklabel objects>)




![png](twitter-model_files/twitter-model_29_1.png)




```python
tweets_hums = tweets_corr[tweets_corr['user_type']==1]
tweets_hums = tweets_corr.drop(['user_type'], axis=1)
plt.matshow(tweets_hums.corr())
plt.title('Correlation Matrix for Humans', y=1.5)
plt.xticks(range(len(tweets_bots.columns)), tweets_bots.columns);
plt.yticks(range(len(tweets_bots.columns)), tweets_bots.columns);
plt.xticks(rotation=90)
```





    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
     <a list of 13 Text xticklabel objects>)




![png](twitter-model_files/twitter-model_30_1.png)




```python


plt.matshow(tweets_corr.corr())
plt.title('Correlation Matrix for Dataset', y=1.5)
plt.xticks(range(len(tweets_bots.columns)), tweets_bots.columns);
plt.yticks(range(len(tweets_bots.columns)), tweets_bots.columns);
plt.xticks(rotation=90)


```





    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
     <a list of 13 Text xticklabel objects>)




![png](twitter-model_files/twitter-model_31_1.png)




```python
def compare_hist(ax, column):
    bins = (max(tweets_hums_re[column]) - min(tweets_hums_re[column]))/15
    plot_title='Histogram of Values for Twitter Bot Detection Dataset Feature: "' + str(column) +'"'
    xlabelval='Values for feature: ' + str(column)
    ax[i].hist(tweets_hums_re[column], label=['human'], alpha=0.5, bins=15)
    ax[i].hist(tweets_bots[column], label=['bot'], color='red', alpha=0.5, bins=15)
    ax[i].set_title(plot_title)
    ax[i].set_xlabel(xlabelval)
    ax[i].set_ylabel('Frequency')
    ax[i].legend()
```




```python
tweets_hums_re = resample(tweets_hums, n_samples=len(tweets_bots), replace=False)
```




```python
fig, ax = plt.subplots(7,2, figsize=(5,2.5))
plt.subplots_adjust(left=2, bottom=None, right=5, top=15, wspace=None, hspace=0.5)
ax = ax.ravel()
i=0
j=0
for column in tweets_hums.columns: 
    j=0
    bins = (max(tweets_hums_re[column]) - min(tweets_hums_re[column]))/15
    plot_title='Histogram of Values for Twitter Bot Detection Dataset Feature: "' + str(column) +'"'
    xlabelval='Values for feature: ' + str(column)
    ax[i].hist(tweets_hums_re[column], label=['human'], alpha=0.5, bins=15)
    ax[i].hist(tweets_bots[column], label=['bot'], color='red', alpha=0.5, bins=15)
    ax[i].set_title(plot_title)
    ax[i].set_xlabel(xlabelval)
    ax[i].set_ylabel('Frequency')
    ax[i].legend()
    i+=1
fig.delaxes(ax[13])   
```


    /anaconda3/lib/python3.6/site-packages/numpy/lib/histograms.py:754: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    /anaconda3/lib/python3.6/site-packages/numpy/lib/histograms.py:755: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)



![png](twitter-model_files/twitter-model_34_1.png)




```python
fig, ax = plt.subplots(7,2, figsize=(5,2.5))
plt.subplots_adjust(left=2, bottom=None, right=5, top=15, wspace=None, hspace=0.5)
ax = ax.ravel()
i=0
j=0
for column in tweets_hums.columns: 
    j=0
    plot_title='Scatter Matrix of Values for Twitter Bot Detection Dataset Feature: "' + str(column) +'"'
    xlabelval='Values for Human feature: ' + str(column)
    ylabelval='Values for Bot feature: ' + str(column)
    ax[i].scatter(tweets_hums_re[column], tweets_bots[column],  alpha=0.5)
    ax[i].set_title(plot_title)
    ax[i].set_xlabel(xlabelval)
    ax[i].set_ylabel(ylabelval)
    ax[i].legend()
    i+=1
fig.delaxes(ax[13])   
```



![png](twitter-model_files/twitter-model_35_0.png)


### Resources


https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/

https://codereview.stackexchange.com/questions/181152/identify-and-extract-urls-from-text-corpus

https://en.wikipedia.org/wiki/Naive_Bayes_classifier

https://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/

https://nlpforhackers.io/topic-modeling/

https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184

http://www.nltk.org/book/ch06.html

https://medium.freecodecamp.org/basic-data-analysis-on-twitter-with-python-251c2a85062e 


# importing the libraries 

import numpy as np 
import string 
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_sentence(sentence) :
    """
    
    Parameters
    ----------
    sentence : a string of words 

    Returns
    -------
    clean_sentence : a string of words without having the unessasry words 

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    sentence = re.sub(r'\$\w*', '', sentence)
    # remove old style retweet text "RT"
    sentence = re.sub(r'^RT[\s]+', '', sentence)
    # remove hyperlinks
    sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
    # remove hashtags
    # only removing the hash # sign from the word
    sentence = re.sub(r'#', '', sentence)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    
    sentence_tokens = tokenizer.tokenize(sentence)
    
    clean_sentence = []
    for word in sentence_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            
            stem_word = stemmer.stem(word)  # stemming word
            clean_sentence.append(stem_word)
    
    return clean_sentence

def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of sentences
        ys: an m x 1 array with the sentiment label of each sentence
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_sentence(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
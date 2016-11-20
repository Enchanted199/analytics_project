# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from operator import itemgetter
import pandas as pd
import re


def clean_essay(string):
    string = re.sub(r"\\t", " ", string)   
    string = re.sub(r"\\n", " ", string)   
    string = re.sub(r"\\r", " ", string)   
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    return string.strip()


def prep_data(df, text_vars):
    for var in text_vars:
            df[var][pd.isnull(df[var])] = ""
            df[var] = df[var].apply(clean_essay)


def create_texts(df, text_vars, stem=False):
    pattern = r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b"
    tokenizer = RegexpTokenizer(pattern)
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    
    texts = []
    for index, row in df.iterrows():
        all_tokens = []
        for var in text_vars:
            tokens = tokenizer.tokenize(row[var])
        
            # remove stop words from tokens
            tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            if stem:
                tokens = [p_stemmer.stem(i) for i in tokens]
        
            all_tokens += tokens
        # add tokens to list
        texts.append(all_tokens)
    
    return texts


def create_texts_and_dictionary(df, text_vars, stem=False):
    texts = create_texts(df, text_vars)
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.05, keep_n=100000)
    dictionary.compactify()
    return (texts, dictionary)


def create_doc_vectors(df, text_vars, dictionary, stem=False, idfield='projectid'):
    pattern = r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b"
    tokenizer = RegexpTokenizer(pattern)
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    
    vectors = []
    for index, row in df.iterrows():
        all_tokens = []
        for var in text_vars:
            tokens = tokenizer.tokenize(row[var])
        
            # remove stop words from tokens
            tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            if stem:
                tokens = [p_stemmer.stem(i) for i in tokens]
        
            all_tokens += tokens
        # add tuple id and bag of words
        doc = dictionary.doc2bow(all_tokens)
        vectors.append((row[idfield], doc))
    
    return vectors
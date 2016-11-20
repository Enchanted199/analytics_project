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
import gensim
import pandas as pd
import os
import re
import argparse

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


"""
Start Script
"""
def main(args):
    if not os.path.exists(args.inputfile):
        raise Exception("Input file does not exist!: " + args.inputfile)

    df = pd.read_csv(args.inputfile)
    print("preparing data...")
    prep_data(df, args.fields)
    
    print("Creating text and loading dictionary for " + str(args.fields) + "...")
    dictionary = corpora.Dictionary().load_from_text(args.dictfile)
    
    ldamodel = gensim.models.ldamodel.LdaModel.load(args.modelfile)
    
    texts = create_texts(df, args.fields, args.dostem)
    for text in texts[0:10]:
        doc_topics = ldamodel[dictionary.doc2bow(text)]
        _srtd = sorted(doc_topics, key=itemgetter(1), reverse=1)
        print(text)
        top, score = _srtd[0]
        print(ldamodel.print_topic(top))
        print("--------------------------------")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="input data file", required=True)
    parser.add_argument("-m", "--modelfile", help="input lda model file", required=True)
    parser.add_argument("-d", "--dictfile", help="input dictonary file", required=True)
    parser.add_argument("-f", "--fields", nargs='+', help="fields to generate models for", required=True)
    parser.add_argument("--dostem", help="should terms be stemmed, default is no stemming",
                        required=False, action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    

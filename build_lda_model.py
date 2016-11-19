# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas as pd
import os
import re
import time
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

def create_dictionary(df, text_vars, stem=False):
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
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.05, keep_n=100000)
    dictionary.compactify()
    return (texts, dictionary)


"""
Start Script
"""
def main(args):
    if not os.path.exists(args.inputfile):
        raise Exception("Input file does not exist!: " + args.inputfile)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    df = pd.read_csv(args.inputfile)
    print("preparing data...")
    prep_data(df, args.fields)
    
    start = time.time()
    print("Creating text and dictionary for " + str(args.fields) + "...")
    texts, dictionary = create_dictionary(df, args.fields, args.dostem)
    dict_file_name = args.outdir + "/" + args.prefix + '-dict.txt'
    print("Saving dictionary to " + dict_file_name)
    dictionary.save_as_text(dict_file_name)
    
    print("Createing corpus for " + str(args.fields) + "...")
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("Generating ldamodel for " + str(args.fields) + "...")
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=args.num_topics, 
                                               id2word = dictionary, passes=args.passes)
    if args.printmodel:
        print(ldamodel.print_topics(num_topics=args.num_topics, num_words=10))

    lda_model_file = args.outdir + "/" + args.prefix + '-lda.model'
    print("Saving ldamodel to " + lda_model_file)
    ldamodel.save(lda_model_file)
    print("Total processing time " + str(time.time() - start) + " for " + str(args.fields))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="input csv file", required=True)
    parser.add_argument("-o", "--outdir", help="where to store results", required=True)
    parser.add_argument("-f", "--fields", nargs='+', help="fields to generate models for", required=True)
    parser.add_argument("-p", "--prefix", default="text", help="prefix for naming saved data", required=False)
    parser.add_argument("--printmodel", help="print out the topics", required=False, action="store_true")
    parser.add_argument("--dostem", help="should terms be stemmed, default is no stemming",
                        required=False, action="store_true", default=False)
    parser.add_argument("-n", "--num_topics", help="number of topics to build model with", 
                        required=False, type=int, default=50)
    parser.add_argument("--passes", help="number of passes against data to build model with", 
                        required=False, default=10)
    args = parser.parse_args()
    main(args)
    
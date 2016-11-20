# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""

from gensim import corpora, models
from operator import itemgetter
import gensim
import pandas as pd
import os
import argparse
import text_util


"""
Start Script
"""
def main(args):
    if not os.path.exists(args.inputfile):
        raise Exception("Input file does not exist!: " + args.inputfile)

    df = pd.read_csv(args.inputfile)
    print("preparing data...")
    text_util.prep_data(df, args.fields)
    
    print("Creating text and loading dictionary for " + str(args.fields) + "...")
    dictionary = corpora.Dictionary().load_from_text(args.dictfile)
    
    ldamodel = gensim.models.ldamodel.LdaModel.load(args.modelfile)
    
    texts = text_util.create_texts(df, args.fields, args.dostem)
    counts = {}
    for text in texts:
        doc = dictionary.doc2bow(text)
        doc_topics = ldamodel[doc]
        _srtd = sorted(doc_topics, key=itemgetter(1), reverse=1)
        top, score = _srtd[0]
        if top in counts:
            counts[top] += 1
        else:
            counts[top] = 1
            
    print(str(counts))

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
    

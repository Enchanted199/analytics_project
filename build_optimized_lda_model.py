# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""

import gensim
import pandas as pd
import os
import time
import argparse
import text_util
import sys
import convert_to_features

"""
Start Script
"""
def main(args):
    if not os.path.exists(args.inputfile):
        raise Exception("Input file does not exist!: " + args.inputfile)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    start = time.time()
    df = pd.read_csv(args.inputfile)

    print("preparing data...")
    text_util.prep_data(df, args.fields)
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    
    print("Creating text and dictionary for " + str(args.fields) + "...")
    dictionary = text_util.create_iterative_dictionary(df, args.fields, args.dostem,
                                                       args.no_below, args.no_above)
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    
    dict_file_name = args.outdir + "/" + args.prefix + '-opt-dict.txt'
    print("Saving dictionary to " + dict_file_name)
    dictionary.save_as_text(dict_file_name)
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    
    tokenizer, en_stop, stemmer = text_util.get_tokenizer_tools()
    print("Createing corpus for " + str(args.fields) + "...")
    
    # convert tokenized documents into a document-term matrix
    corpus = []
    ids = []
    for index, row in df.iterrows():
        all_tokens = text_util.tokenize_row(row, args.fields, tokenizer, en_stop, stemmer, args.dostem)
        corpus.append(dictionary.doc2bow(all_tokens))
        ids.append(row['projectid'])
        if len(corpus) % 1000 == 0:
            print("time: " + str(time.time() - start) + " count: " + str(len(corpus)))
            sys.stdout.flush()

    print("Generating ldamodel for " + str(args.fields) + "...")
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=args.num_topics, 
                                               id2word = dictionary, passes=args.passes)

    if args.printmodel:
        print(ldamodel.print_topics(num_topics=args.num_topics, num_words=10))

    lda_model_file = args.outdir + "/" + args.prefix + '-opt-lda.model'
    print("Saving ldamodel to " + lda_model_file)
    ldamodel.save(lda_model_file)
    print("Total processing time " + str(time.time() - start) + " for " + str(args.fields))
    if args.convertinput:
        print("Converting document vectors to topic features...")
        topic_csv = args.outdir + "/" + args.prefix + '-topics.csv'
        convert_to_features.convert(list(zip(ids, corpus)), ldamodel, topic_csv)

    print_config(args)


def print_config(args):
    print("Input: " + args.inputfile)
    print("Output: " + args.outdir)
    print("Model Settings: ")
    print("  Do Stem?:" + str(args.dostem))
    print("  No Above: " + str(args.no_above))
    print("  No Below: " + str(args.no_below))
    print("  Num Topics: " + str(args.num_topics))
    print("  Passes: " + str(args.passes))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="input csv file", required=True)
    parser.add_argument("-o", "--outdir", help="where to store results", required=True)
    parser.add_argument("-f", "--fields", nargs='+', help="fields to generate models for", required=True)
    parser.add_argument("-p", "--prefix", default="text", help="prefix for naming saved data", required=False)
    parser.add_argument("--printmodel", help="print out the topics", required=False, action="store_true")
    parser.add_argument("--dostem", help="should terms be stemmed, default is no stemming",
                        required=False, action="store_true", default=False)
    parser.add_argument("--convertinput", help="should csv with topic features be created from input documents",
                        required=False, action="store_true", default=False)
    parser.add_argument("-n", "--num_topics", help="number of topics to build model with", 
                        required=False, type=int, default=50)
    parser.add_argument("--passes", help="number of passes against data to build model with", 
                        required=False, default=10)
    parser.add_argument("--no_below", help="minimum documents token must appear", 
                        required=False, type=int, default=5)
    parser.add_argument("--no_above", help="tokens that appearn in this percentage of documents are pruned", 
                        required=False, type=float, default=0.20)
    args = parser.parse_args()
    main(args)
    
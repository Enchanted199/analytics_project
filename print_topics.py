# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:59:51 2016

@author: bbux
"""
import sys

import gensim

if __name__ == "__main__":
    file = sys.argv[1]
    n = int(sys.argv[2])
    ldamodel = gensim.models.ldamodel.LdaModel.load(file)
    print(ldamodel.print_topics(num_topics=n, num_words=10))
from bm25_RF import bm25_RF
from nltk import tokenize
from numpy import bmat, exp
import query_preprocessing
import bm25_basic
import bm25_RF

def generate_recommendations(query, number):
    tokenQuery = query_preprocessing.process_query(query)
    expandedQuery = bm25_basic.query_expansion(tokenQuery)
    bm25Basic_recommendations = bm25_basic.bm25_basic(expandedQuery, n=number)
    bm25RF_recommendations = bm25_RF.bm25_RF(expandedQuery, n=number)
    return bm25Basic_recommendations, bm25RF_recommendations

#print(generate_recommendations("lovely day great energy", 5))
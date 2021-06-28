from nltk import tokenize
from numpy import bmat, exp
import query_preprocessing
import bm25_basic

def generate_recommendations(query, number):
    tokenQuery = query_preprocessing.process_query(query)
    expandedQuery = bm25_basic.query_expansion(tokenQuery)
    bm25Basic_recommendations = bm25_basic.bm25_basic(expandedQuery, n=number)
    return bm25Basic_recommendations


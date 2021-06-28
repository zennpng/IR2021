from nltk import tokenize
from numpy import exp
import query_preprocessing
import bm25_basic

query = "party all night"

tokenQuery = query_preprocessing.process_query(query)
expandedQuery = bm25_basic.query_expansion(tokenQuery)
bm25Basic_recommendations = bm25_basic.bm25_basic(expandedQuery, n=10)

print(bm25Basic_recommendations)

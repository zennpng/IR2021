from bm25_RF import bm25_RF
from nltk import tokenize
from numpy import bmat, exp
import query_preprocessing
import VSM1_1
import bm25_basic
import bm25_RF

def generate_recommendations(query, number):
    tokenQuery = query_preprocessing.process_query(query)
    expandedQuery = query_preprocessing.query_expansion(tokenQuery)
    vsm1_1_recommendations, sortID, prodlist = VSM1_1.type_of_vsm(expandedQuery, method = "dotprod", vsm_type = 1, n=number) 
    bm25Basic_recommendations = bm25_basic.bm25_basic(expandedQuery, n=number)
    bm25RF_recommendations = bm25_RF.bm25_RF(expandedQuery, n=number)
    return vsm1_1_recommendations, bm25Basic_recommendations, bm25RF_recommendations

#print(generate_recommendations("lovely day great energy", 5))
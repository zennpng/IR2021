import query_preprocessing
import VSM1_1
import bm25_full
import LangModel
import gensim.downloader as api


def generate_recommendations(query, number):
    tokenQuery = query_preprocessing.process_query(query)
    expandedQuery = query_preprocessing.query_expansion(tokenQuery)
    wv = api.load('word2vec-google-news-300')
    vsm1_1_recommendations, sortID, prodlist = VSM1_1.type_of_vsm(wv, expandedQuery, method = "dotprod", vsm_type=1, n=number) 
    return vsm1_1_recommendations

print(generate_recommendations("happy and joyful", 4))

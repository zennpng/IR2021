import query_preprocessing
import VSM1_1
import bm25_full
import LangModel
import gensim.downloader as api


def generate_recommendations(query, number):
    tokenQuery = query_preprocessing.process_query(query)
    expandedQuery = query_preprocessing.query_expansion(tokenQuery)
    vsm1_1_recommendations = VSM1_1.type_of_vsm(wv, expandedQuery, vsm_type=1, method="cosine", n=number) 
    return vsm1_1_recommendations

wv = api.load('word2vec-google-news-300')
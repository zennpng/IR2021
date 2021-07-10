import query_preprocessing
import VSM1_1
import bm25_full

def generate_recommendations(query, number):
    tokenQuery = query_preprocessing.process_query(query)
    expandedQuery = query_preprocessing.query_expansion(tokenQuery)
    vsm1_1_recommendations, sortID, prodlist = VSM1_1.type_of_vsm(expandedQuery, method = "dotprod", vsm_type = 1, n=number) 
    selected_docsID, bm25_recommendations = bm25_full.bm25(expandedQuery, n=number)
    return vsm1_1_recommendations, bm25_recommendations

#print(generate_recommendations("lovely day great energy", 5))
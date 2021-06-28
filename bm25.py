import pandas as pd
from tqdm import tqdm
from itertools import chain
from nltk.corpus import wordnet as wn
import pickle
import math 

# Query expansion 
def query_expansion(query_list):
    expanded_query = []
    for word in query_list:
        synonyms = wn.synsets(word)
        expansion = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        expansion = [ x for x in expansion if "_" not in x ]
        expanded_query += expansion
    return(expanded_query)

query = query_expansion(["workout", "energy"])
# print(query)
# ['exercise', 'workout', 'exercising', 'energy', 'zip', 'vim', 'Energy', 'vigour', 'vigor', 'vitality', 'muscularity', 'get-up-and-go', 'push', 'DOE']


## BM25 (Basic Version)

with open('BM25_index.txt','rb') as handle:
    indexes = pickle.loads(handle.read())
#print(indexes["holiday"])

def _score(query, doc_id, docs, index, k1=1.5, b=0.75):
    score = 0.0
    corpus_size = 28372
    avg_doc_len = 0 
    for d in docs:
        avg_doc_len += len(d)
    avg_doc_len = avg_doc_len/len(docs)
    for term in query:
        if term not in index.keys():
            continue
        df = index[term]["doc_freq"] 
        idf = round(math.log((corpus_size)/(df)),2)
        tf = 0 
        for tfset in index[term]["doc_list"]:
            if tfset[0] == doc_id:
                tf = tfset[1]
        doc_len = len(docs[doc_id-1])
        score += idf*((k1+1)*tf)/(k1*((1-b)+b*(doc_len/avg_doc_len))+tf)
    return score

raw_df = pd.read_csv("tcc_ceds_music.csv")

corpus = [] 
for lyric in raw_df['lyrics']:
    corpus.append(lyric)

scores = []
doc_id_list = list(range(1,28373))
for doc_id in tqdm(range(1,28373)):
    scores.append(_score(query, doc_id, corpus, indexes))
#print(scores)
sorted_scores = sorted(scores, reverse=True)
#print(sorted_scores)
sorted_docsID = [x for _, x in sorted(zip(scores, doc_id_list), reverse=True)]
#print(sorted_docsID)

# limit no of recommendations
n = 5
recommended_docs = sorted_docsID[0:n]
print(recommended_docs)
    

## BM25 (Relevant Feedback Version)




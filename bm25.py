import pandas as pd
from tqdm import tqdm

## BM25 Algo 

def tf_(doc):
    frequencies= {}
    for letter in doc:
        if letter in frequencies:
            frequencies[letter] += 1
        else:
            frequencies[letter] = 1
        
    return frequencies

#input = ["a","b","b","c","d"]
#print(tf_(input))  {'a': 1, 'b': 2, 'c': 1, 'd': 1}

def df_(docs):
    df = {}
    for doc in docs:
        for term in set(doc):
            df_count = df.get(term, 0) + 1
            df[term] = df_count
    return df

#input = [['a','b','c'],['b','c','d'],['c','d','e']]
#print(df_(input))  {'a': 1, 'b': 2, 'c': 3, 'd': 2, 'e': 1}

def idf_(df, corpus_size):
    import math 
    idf = {}
    for term, freq in df.items():
        idf[term] = round(math.log((corpus_size)/(freq)),2)
    return idf

def _score(query, doc, docs, k1=1.5, b=0.75):
    score = 0.0
    tf = tf_(doc)
    df = df_(docs)
    idf = idf_(df, len(docs))
    avg_doc_len = 0 
    for d in docs:
        avg_doc_len += len(d)
    avg_doc_len = avg_doc_len/len(docs)
    for term in query:
        if term not in tf.keys():
            continue
        score += idf[term]*((k1+1)*tf[term])/(k1*((1-b)+b*(len(doc)/avg_doc_len))+tf[term])
    return score

# test data 
query = "sun is shining what a wonderful day"
#corpus = ['hello there good man', 'it is really quite windy in london', 'how is the weather in london today']

# actual data preprocessing
raw_df = pd.read_csv("tcc_ceds_music.csv")
raw_df = raw_df.iloc[: , 1:]
#print(raw_df.head())
#print(raw_df['lyrics'][0:5])

corpus = [] 
for lyric in raw_df['lyrics'][-1000:]:
    corpus.append(lyric)
#print(len(corpus))

tokenized_query = query.split(" ")
tokenized_corpus = [doc.split(" ") for doc in corpus]
scores = []
for doc in tqdm(tokenized_corpus):
    scores.append(_score(tokenized_query,doc,tokenized_corpus))
#print(scores)
sorted_scores = sorted(scores, reverse=True)
#print(sorted_scores)
sorted_docs = [x for _, x in sorted(zip(scores, corpus), reverse=True)]
#print(sorted_docs)

# limit no of recommendations
n = 5
recommended_docs = sorted_docs[0:n]
print(recommended_docs)
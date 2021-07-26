import pandas as pd
from tqdm import tqdm
import pickle
import math
import multiprocessing as mp 

def _score(query, doc_id, docs,avg_doc_len, index, k1=1.5, b=0.75):
        score = 0.0
        corpus_size = 28372
        # avg_doc_len = 0 
        # for d in docs:
        #     avg_doc_len += len(d)
        # avg_doc_len = avg_doc_len/len(docs)
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

## BM25 (Basic Version with multi processing)
def bm25_basic(query, n=5):
    
    # load index for bm25
    with open('BM25_index.txt','rb') as handle:
        indexes = pickle.loads(handle.read())

    music_df = pd.read_csv("tcc_ceds_music.csv")

    corpus = []
    avg_doc_len = 0 
    for lyric in music_df['lyrics']:
        corpus.append(lyric)
        avg_doc_len += len(lyric)
    avg_doc_len = avg_doc_len/len(corpus)

    doc_id_list = list(range(1,28373))

    # # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())   
    
    print("Scoring documents...")
    # Step 2: `pool.starmap` the `_scores`
    # https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    # can play with additional chunksize argument to see if it processes faster, use 28373/8(no. of processors)
    scores = pool.starmap(_score, tqdm([(query, doc_id, corpus,avg_doc_len, indexes) for doc_id in range(1,28373)]))

    # Step 3: Don't forget to close
    pool.close()
    
    sorted_scores = sorted(scores, reverse=True)
    sorted_docsID = [x for _, x in sorted(zip(scores, doc_id_list), reverse=True)]

    # get recommended song infos
    recommended_song_infos = []
    selected_docsID = sorted_docsID[0:n]
    for docID in selected_docsID:
        recommended_song_infos.append(music_df['artist_name'][docID-1] + " - " + music_df['track_name'][docID-1] + ", " + str(music_df['release_date'][docID-1]) + " " + music_df['genre'][docID-1])
    return selected_docsID, recommended_song_infos
    





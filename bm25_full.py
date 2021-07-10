import pandas as pd
from tqdm import tqdm
import pickle
import math 
import ast


## BM25 Full Version (Basic + RF)
def bm25(query, n=5):
    with open('BM25_index.txt','rb') as handle:
        indexes = pickle.loads(handle.read())

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

    music_df = pd.read_csv("tcc_ceds_music.csv")

    corpus = [] 
    for lyric in music_df['lyrics']:
        corpus.append(lyric)

    scores = []
    doc_id_list = list(range(1,28373))
    for doc_id in tqdm(range(1,28373)):
        scores.append(_score(query, doc_id, corpus, indexes))
    sorted_scores = sorted(scores, reverse=True)
    sorted_docsID = [x for _, x in sorted(zip(scores, doc_id_list), reverse=True)]

    # get recommended song infos
    selected_docsID = sorted_docsID[0:n]
    file = open("bm25_relevant.txt", "r")
    old_relevant = ast.literal_eval(file.read())
    new_relevant = list(set(old_relevant + selected_docsID))
    with open('bm25_relevant.txt', 'w') as f:
        f.write(str(new_relevant))

    # relevance feedback implementation (if relevant docs > 50 and is at intervals of 5)

    if (len(old_relevant) > 50) and (len(old_relevant)%5==0):
        relevant_retrievedDocs = list(set(selected_docsID).intersection(old_relevant))
        vr = sum(dID in selected_docsID for dID in old_relevant)

        def _updatedScore(query, doc_id, docs, index, k1=1.5, b=0.75):
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
                vr_t = 0
                for rdoc in relevant_retrievedDocs:
                    termdocmap = index[term]["doc_list"]
                    if [doc for doc in termdocmap if doc[0] == rdoc] != []:
                        vr_t += 1
                vnr_t = vr - vr_t
                idf_rf = ((vr_t+0.5)/(vnr_t+0.5))/((df-vr_t+0.5)/(corpus_size-df-vr+vr_t+0.5))
                tf = 0 
                for tfset in index[term]["doc_list"]:
                    if tfset[0] == doc_id:
                        tf = tfset[1]
                doc_len = len(docs[doc_id-1])
                score += round(math.log(idf_rf*((k1+1)*(tf+0.1))/(k1*((1-b)+b*(doc_len/avg_doc_len))+tf)),2)
            return score

        updatedScores = []
        for doc_id in tqdm(range(1,28373)):
            updatedScores.append(_updatedScore(query, doc_id, corpus, indexes))
        updatedSorted_scores = sorted(updatedScores, reverse=True)
        updatedSorted_docsID = [x for _, x in sorted(zip(updatedScores, doc_id_list), reverse=True)]
        selected_docsID = updatedSorted_docsID[0:n]
        
    recommended_song_infos = []
    for docID in selected_docsID:
        recommended_song_infos.append(music_df['artist_name'][docID-1] + " - " + music_df['track_name'][docID-1] + ", " + str(music_df['release_date'][docID-1]) + " " + music_df['genre'][docID-1])
    return selected_docsID, recommended_song_infos
    





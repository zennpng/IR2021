1# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:30:57 2021

@author: teresa
"""
import pandas as pd
from tqdm import tqdm
import pickle
import query_preprocessing
import scipy as sp
import numpy as np

#28372 songs
#each vector has 300 attributes
#music_df = pd.read_csv("tcc_ceds_music.csv")
#with open('VSM1_1_index.txt', 'rb') as handle:
   #file=pickle.loads(handle.read())
   #print(file)
    

#query: list of words after query expansion
#method: to measure similairity
query = ["miss", "family"] 


#VSM

def type_of_vsm(wv, query, vsm_type=1, method="cosine", n=5):
    music_df = pd.read_csv("tcc_ceds_music.csv")
    
    if vsm_type == 1:
        with open('VSM1_1_index.txt', 'rb') as handle:
            file=pickle.loads(handle.read())
    else:
        with open('VSM1_2_index.txt', 'rb') as handle:
            file=pickle.loads(handle.read())
        

    def vsm(query, method, n, wv):
        
            # import gensim.downloader as api
            # wv = api.load('word2vec-google-news-300')
            word_vec_dict = {}
            query_vec = 0
            for i in range(len(query)):
                if query[i] in wv:
                    
                    word_vec_dict[query[i]] = wv[query[i]]
                    query_vec = query_vec + word_vec_dict[query[i]]
    
            
            doc_id_list = list(range(1,28373))
            
            
            
            if method == "cosine":
                sim_list = []
                for key, value in tqdm(file.items()):
                    cos_sim = 1 - sp.spatial.distance.cosine(value, query_vec)
                    sim_list.append(cos_sim)
                 #sort from highest value to lowest  
                sorted_ID = [x for _, x in sorted(zip(sim_list, doc_id_list), reverse=True)]
                
            if method == "euclid":
                e_dist_list = []
                for key, value in tqdm(file.items()):
                    dist = np.linalg.norm(value-query_vec)
                    e_dist_list.append(dist)
                #sort from lowest value to highest
                sorted_ID = [x for _, x in sorted(zip(e_dist_list, doc_id_list))]
            
            if method == "dotprod":
                prod_list =[]
                for key, value in tqdm(file.items()):
                    prod = np.inner(value, query_vec)
                    prod_list.append(prod)
                    
                #sort from highest to lowest value
                sorted_ID = [x for _, x in sorted(zip(prod_list, doc_id_list), reverse=True)]
                
            return(sorted_ID[:n], prod_list[:n])   
    
    sorted_ID_final, prod_list_final = vsm(query, method, n, wv)            
    recommended_song_infos = []
    for docID in sorted_ID_final[0:n]:
        recommended_song_infos.append(music_df['artist_name'][docID-1] + " - " + music_df['track_name'][docID-1] + ", " + str(music_df['release_date'][docID-1]) + " " + music_df['genre'][docID-1])
                
        
    return(recommended_song_infos,sorted_ID_final[:n], prod_list_final)
            
def type_of_vsm_old(query, vsm_type=1, method="cosine", n=5):
    music_df = pd.read_csv("tcc_ceds_music.csv")
    
    if vsm_type == 1:
        with open('VSM1_1_index.txt', 'rb') as handle:
            file=pickle.loads(handle.read())
    else:
        with open('VSM1_2_index.txt', 'rb') as handle:
            file=pickle.loads(handle.read())
        

    def vsm(query, method, n):
        
            import gensim.downloader as api
            wv = api.load('word2vec-google-news-300')
            word_vec_dict = {}
            query_vec = 0
            for i in range(len(query)):
                if query[i] in wv:
                    
                    word_vec_dict[query[i]] = wv[query[i]]
                    query_vec = query_vec + word_vec_dict[query[i]]
    
            
            doc_id_list = list(range(1,28373))
            
            
            
            if method == "cosine":
                sim_list = []
                for key, value in tqdm(file.items()):
                    cos_sim = 1 - sp.spatial.distance.cosine(value, query_vec)
                    sim_list.append(cos_sim)
                 #sort from highest value to lowest  
                sorted_ID = [x for _, x in sorted(zip(sim_list, doc_id_list), reverse=True)]
                
            if method == "euclid":
                e_dist_list = []
                for key, value in tqdm(file.items()):
                    dist = np.linalg.norm(value-query_vec)
                    e_dist_list.append(dist)
                #sort from lowest value to highest
                sorted_ID = [x for _, x in sorted(zip(e_dist_list, doc_id_list))]
            
            if method == "dotprod":
                prod_list =[]
                for key, value in tqdm(file.items()):
                    prod = np.inner(value, query_vec)
                    prod_list.append(prod)
                    
                #sort from highest to lowest value
                sorted_ID = [x for _, x in sorted(zip(prod_list, doc_id_list), reverse=True)]
                
            return(sorted_ID[:n], prod_list[:n])   
    
    sorted_ID_final, prod_list_final = vsm(query, method, n)            
    recommended_song_infos = []
    for docID in sorted_ID_final[0:n]:
        recommended_song_infos.append(music_df['artist_name'][docID-1] + " - " + music_df['track_name'][docID-1] + ", " + str(music_df['release_date'][docID-1]) + " " + music_df['genre'][docID-1])
                
        
    return(recommended_song_infos,sorted_ID_final[:n], prod_list_final)        
     
        
    
    
    
    

       
        
        
        


    

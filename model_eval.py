import pandas as pd
import numpy as np
import math
import ast

import bm25_basic as bm
import VSM1_1 as vsm
import query_preprocessing

class Evaluator:

    def __init__(self, eval_filepath):
        '''
        init an Evaluator object that stores the evaluation dataset in memory
        '''
        self.testset = pd.read_csv(eval_filepath)
        self.predictset = pd.read_csv(eval_filepath)


    # define some utility functions
    def precision_at_k(self, r, k):
        '''
        r: relevance scores e.g. [0,0,1,0,1,1]
        '''
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)
    

    def average_precision(self, r):
        '''
        Score is average precision (area under PR curve)
        
        r: relevance scores e.g. [0,0,1,0,1,1]

        Returns: Average precision
        '''
        sum_p = 0
        total_relevant = 0
        for k in range(len(r)):
            if r[k] == 1:
                total_relevant += 1
                sum_p += self.precision_at_k(r,k+1)
        
        try:
            avg_p = sum_p/total_relevant
        except:
            avg_p = 0
        return avg_p    

    # performance measure 1
    def mean_average_precision(self, rs):
        '''
        rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0,1]]

        rs: relevance scores for entire test set (each item is results for each song)

        Returns: Mean average precision
        '''
        sum_ave_p = 0
        for r in rs:
            sum_ave_p += self.average_precision(r)
        m_avg_p = sum_ave_p/len(rs)

        return m_avg_p    


    def dcg_at_k(self, r, k):
        '''
        Score is discounted cumulative gain (dcg)
        Relevance is positive real values.  Can use binary as the previous methods.

        r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        k = Number of results to consider

        '''
        dcg = r[0] # first term
        for i in range(k-1):
            dcg += r[i+1] / math.log2(i+2)

        return dcg


    def ndcg_at_k(self, r, k):
        """
        Score is normalized discounted cumulative gain (ndcg)

        Relevance is positive real values.  Can use binary
        as the previous methods.

        r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        k = Number of results to consider

        """
        dcg = self.dcg_at_k(r,k)
        
        # calc ideal ranking idcg, sort rankings by desc order
        r.sort(reverse=True)
        idcg = self.dcg_at_k(r,k)
        try:
            ndcg = dcg/idcg
        except:
            ndcg = 0

        return ndcg

    # performance measure 2
    def mean_ndcg(self,rs,k=10):
        '''
        rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0,1]]

        rs: relevance scores for entire test set (each item is results for each song)

        Returns: Mean NDCG
        '''
        sum_ndcg = 0
        for r in rs:
            sum_ndcg += self.ndcg_at_k(r,k)
        m_ndcg = sum_ndcg/len(rs)
        return m_ndcg

    
    def reciprocal_rank(self,r):
        '''
        r: relevance scores e.g. [0,0,1,0,1,1]

        Returns: reciprocal rank 1/R where R is position of first relevant doc
        '''
        # store flag of whether first relevant doc has been found
        relevant_doc_not_found = True
        
        rank = 0
        # iterate through scores in r
        for i in range(len(r)):
            if r[i] == 1 and relevant_doc_not_found:
                rank = i+1
                relevant_doc_not_found = False
        try:
            return 1/rank
        except:
            return 0
    
    # performance measure 3
    def mean_reciprocal_rank(self,rs):
        '''
        rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0,1]]

        rs: relevance scores for entire test set (each item is results for each song)

        Returns: Mean reciprocal rank across queries
        '''
        sum_rr = 0
        for r in rs:
            sum_rr += self.reciprocal_rank(r)
        m_rr = sum_rr/len(rs)

        return m_rr


    # main functions of this Evaluator class
    def make_predictions(self, model):

        # create and convert predictions column in self.predictset to dtype 'object'
        self.predictset['predictions'] = ''
        self.predictset = self.predictset.astype({'predictions':object})
        
        # for each query in test set, obtain song recommendations from model
        for row in range(len(self.testset)):
            # retrieve test set query
            query = self.testset.loc[row, 'queryString']

            # pre-process the test query
            tokenQuery = query_preprocessing.process_query(query)
            expandedQuery = query_preprocessing.query_expansion(tokenQuery)

            # get model output
            if model == 'bm25_basic':
                selected_docsID, recommended_song_infos = bm.bm25_basic(expandedQuery, 28373)
                self.predictset.at[row,'predictions'] = selected_docsID
                
            
            elif model == 'vsm_1_1':
                recommended_song_infos,sorted_ID_final, prod_list_final = vsm.type_of_vsm(expandedQuery, method = "dotprod", vsm_type = 1, n=28373)
                self.predictset.at[row,'predictions'] = sorted_ID_final
        
        # output predictions
        filename = model + '_predictions.csv'
        self.predictset.to_csv(filename, index=False)


    def evaluate(self, model, n):
        '''
        uses the evaluation dataset to calculate various performance metrics of a model based on first n retrieved docs
        '''
        # load predictions dataset
        filename = model + '_predictions.csv'
        self.predictset = pd.read_csv(filename)
        
        # container for relevance scores of all queries
        rs =[]
        
        # for each query in test set, check relevance of song recommendations from model
        for row in range(len(self.predictset)):

            # get answers (relevant doc for this test query)
            relevant_answers = self.predictset.loc[row, 'songIDs'] # this is a string '[number, number, number]'
            relevant_answers = ast.literal_eval(relevant_answers)

            # get predictions
            predictions = self.predictset.loc[row, 'predictions']
            predictions = ast.literal_eval(predictions) 

            # container for relevance of model outputs for each query
            r = []

            for id in predictions[:n]:
                if id in relevant_answers:
                    r.append(1)
                else:
                    r.append(0)
            
            # store relevance scores of this query into the global container
            rs.append(r)

        # evaluate performance using rs values
        # MAP
        m_avg_p = self.mean_average_precision(rs)

        # mean NDCG across different queries
        m_ndcg = self.mean_ndcg(rs)

        # mean reciprocal rank MRR
        m_rr = self.mean_reciprocal_rank(rs)

        return m_avg_p, m_ndcg, m_rr


### SCRIPT ###
model_eval = Evaluator('test_dataset.csv')
model_eval.make_predictions('bm25_basic')
# m_avg_p, m_ndcg, m_rr = model_eval.evaluate('bm25_basic',30)

# print("###################################")
# print("#   Performance Metric for BM25   #")
# print("###################################")
# print("Mean Average Precision: ", m_avg_p)
# print("Mean NDCG: ", m_ndcg)
# print("Mean Reciprocal Rank: ", m_rr)
# print("-----------------------------------")
# print('\n')

model_eval = Evaluator('test_dataset.csv')
model_eval.make_predictions('vsm_1_1')
# m_avg_p, m_ndcg, m_rr = model_eval.evaluate('vsm_1_1',30)

# print("###################################")
# print("#   Performance Metric for VSM1   #")
# print("###################################")
# print("Mean Average Precision: ", m_avg_p)
# print("Mean NDCG: ", m_ndcg)
# print("Mean Reciprocal Rank: ", m_rr)
# print("-----------------------------------")
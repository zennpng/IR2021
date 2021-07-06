import pandas as pd
import numpy as np
import math

class Evaluator:

    def __init__(self, eval_filepath):
        '''
        init an Evaluator object that stores the evaluation dataset in memory
        '''
        self.testset = pd.read_csv(eval_filepath)


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
        
        avg_p = sum_p/total_relevant
        return avg_p    


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
        ndcg = dcg/idcg

        return ndcg


    # main function of this Evaluator class
    def evaluate(self,model):
        '''
        uses the evaluation dataset to calculate various performance metrics of a model
        '''
        pass


### SCRIPT ###
import pandas as pd

class IndexConstructor:

    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)


### SCRIPT ###
BM25_indexer = IndexConstructor('tcc_ceds_music.csv')
print(BM25_indexer.dataset.head())
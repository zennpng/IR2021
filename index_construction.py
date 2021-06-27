import pandas as pd
import pickle

class IndexConstructor:

    def __init__(self, file_path):
        '''
        init an IndexConstructor object that stores the dataset in memory
        '''
        # since dataset is small enough, we process everything in memory to minimise disk-reads
        self.dataset = pd.read_csv(file_path)
        self.index = {}
    
    
    def index_lyrics(self,index_dd, lyrics, doc_id):
        ''' 
        updates the index stored in self based on lyrics of each song
        
        input: self.index, lyrics, song_id
        output: updates self.index of the form {'word1':{'doc_list':[(d_i,tf_i),...]}, 'word2':{'doc_list':[(d_i,tf_i),...]}} 
        '''
        # tokenise the lyrics
        lyric_words = lyrics.split(" ")

        # temporary storage of term frequencies for this song
        tf = {}

        # parse each word in the lyrics and calculate term freq
        for word in lyric_words:
            if word not in tf.keys():
                tf[word] = 1
            else:
                tf[word] += 1

        # merge the tf into self.index
        for word in tf.keys():
            
            # check if word exists in self.index
            if word not in index_dd.keys():
                # init first entry of word in index with correct format
                index_dd[word] = {'doc_list': [(doc_id,tf[word])]}

            else:
                # update entry in index
                index_dd[word]['doc_list'].append((doc_id,tf[word]))

    
    def calc_df(self,index_dd):
        '''
        calculate and updates the document freq of each term, based on how many documents it appeared in
        
        input: self.index of the form {'word1':{'doc_list':[(d_i,tf_i),...]}, 'word2':{'doc_list':[(d_i,tf_i),...]}}
        output: updates self.index of the form {'word1':{'doc_list':[(d_i,tf_i),...], 'doc_freq': int}, ...}
        '''
        # loop through each word in the index (vocabulary)
        for word in index_dd.keys():
            # retrieve the list of documents where the word appears
            doc_ls = index_dd[word]['doc_list']

            # add a new field to record the doc freq of each word
            index_dd[word]['doc_freq'] = len(doc_ls)
            
    
    def construct_index(self, for_model):
        '''
        construct the index needed for respective models
        '''
        # check which model we are constructing the index for
        if for_model == 'BM25':
            
            # iterate through each song (document) in the dataset
            for row in range(len(self.dataset)):    
                # retrieve the id and lyrics for this song
                doc_id = self.dataset.loc[row,'doc_id']
                lyrics = self.dataset.loc[row,'lyrics']

                # call utility function to update the index
                self.index_lyrics(self.index, lyrics, doc_id)
            
            # parse thru index list to calculate document freq of each term
            self.calc_df(self.index)

        elif for_model == 'VSM':
            pass

    def output(self, file_path):
        '''
        make use of pickle module to output index into a text file

        given dd = {'a': 1, 'b': 2}
        
        with open('file.txt','wb') as handle:
            pickle.dump(dd, handle)

        with open('file.txt','rb') as handle:
            something = pickle.loads(handle.read())
        '''
        # output the index into a file
        with open(file_path,'wb') as file:
            pickle.dump(self.index, file)


### SCRIPT ###
BM25_indexer = IndexConstructor('tcc_ceds_music.csv')
print("Initialized Index Constructor for BM25")
print("Constructing index...")

BM25_indexer.construct_index('BM25')
print("Constructed index for BM25")

BM25_indexer.output('BM25_index.txt')
print("BM25 index saved successfully in the format: ", "{'word1':{'doc_list':[(d_i,tf_i),...], 'doc_freq': int}, ...}")

print(len(BM25_indexer.index.keys()), 'words')
import pandas as pd
import pickle
from tqdm import tqdm

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
            
    
    def make_vectors(self):
        '''
        returns a dictionary of vector embeddings for each attribute (used to describe the songs in the dataset)
        '''
        # ref: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
        # load pre-trained word2vec encoder
        import gensim.downloader as api
        wv = api.load('word2vec-google-news-300')

        # calculate the vectors of each attribute in the dataset
        attr_vec = {}

        attr_vec['dating'] = wv['dating']
        attr_vec['violence'] = wv['violence']
        attr_vec['worldlife'] = wv['world'] + wv['life']
        attr_vec['nighttime'] = wv['night'] + wv['time']
        attr_vec['shaketheaudience'] = wv['shake'] + wv['audience']
        attr_vec['familygospel'] = wv['family'] + wv['gospel']
        attr_vec['romantic'] = wv['romantic']
        attr_vec['communication'] = wv['communication']
        attr_vec['obscene'] = wv['obscene']
        attr_vec['music'] = wv['music']
        attr_vec['movementplaces'] = wv['movement'] + wv['places']
        attr_vec['lightvisual'] = wv['light'] + wv['visual']
        attr_vec['familyspiritual'] = wv['family'] + wv['spiritual']
        attr_vec['likegirls'] = wv['like'] + wv['girls']
        attr_vec['sadness'] = wv['sadness']
        attr_vec['feelings'] = wv['feelings']
        attr_vec['danceability'] = wv['danceability']
        attr_vec['loudness'] = wv['loudness']
        attr_vec['acousticness'] = wv['acoustic']
        attr_vec['instrumentalness'] = wv['instrumental']
        attr_vec['valence'] = wv['valence']
        attr_vec['energy'] = wv['energy']

        return attr_vec

    
    def construct_index(self, for_model):
        '''
        construct the index needed for respective models
        '''
        # check which model we are constructing the index for
        if for_model == 'BM25':
            
            # iterate through each song (document) in the dataset
            for row in tqdm(range(len(self.dataset))):    
                # retrieve the id and lyrics for this song
                doc_id = self.dataset.loc[row,'doc_id']
                lyrics = self.dataset.loc[row,'lyrics']

                # call utility function to update the index
                self.index_lyrics(self.index, lyrics, doc_id)
            
            # parse thru index list to calculate document freq of each term
            self.calc_df(self.index)

        elif for_model == 'VSM1_1':

            # obtain vector representations of song attributes
            attr_vec = self.make_vectors()

            for row in tqdm(range(len(self.dataset))):

                # retrieve the id and attribute values for this song
                doc_id = self.dataset.loc[row,'doc_id']
                weight_dating = self.dataset.loc[row,'dating']
                weight_violence = self.dataset.loc[row,'violence']
                weight_worldlife = self.dataset.loc[row,'world/life']
                weight_nighttime = self.dataset.loc[row,'night/time']
                weight_shaketheaudience = self.dataset.loc[row,'shake the audience']
                weight_familygospel = self.dataset.loc[row, 'family/gospel']
                weight_romantic = self.dataset.loc[row, 'romantic']
                weight_communication = self.dataset.loc[row,'communication']
                weight_obscene = self.dataset.loc[row,'obscene']
                weight_music = self.dataset.loc[row,'music']
                weight_movementplaces = self.dataset.loc[row,'movement/places']
                weight_lightvisual = self.dataset.loc[row,'light/visual perceptions']
                weight_familyspiritual = self.dataset.loc[row,'family/spiritual']
                weight_likegirls = self.dataset.loc[row,'like/girls']
                weight_sadness = self.dataset.loc[row, 'sadness']
                weight_feelings = self.dataset.loc[row,'feelings']
                weight_danceability = self.dataset.loc[row,'danceability']
                weight_loudness = self.dataset.loc[row,'loudness']
                weight_acousticness = self.dataset.loc[row, 'acousticness']
                weight_instrumental = self.dataset.loc[row,'instrumentalness']
                weight_valence = self.dataset.loc[row, 'valence']
                weight_energy = self.dataset.loc[row,'energy']

                # calculate the document vector by summing weighted attr vectors
                doc_vec = weight_dating*attr_vec['dating'] + \
                        weight_violence*attr_vec['violence'] + \
                        weight_worldlife*attr_vec['worldlife'] + \
                        weight_nighttime*attr_vec['nighttime'] + \
                        weight_shaketheaudience*attr_vec['shaketheaudience'] +\
                        weight_familygospel*attr_vec['familygospel'] +\
                        weight_romantic*attr_vec['romantic'] +\
                        weight_communication*attr_vec['communication'] +\
                        weight_obscene*attr_vec['obscene'] +\
                        weight_music*attr_vec['music']+\
                        weight_movementplaces*attr_vec['movementplaces']+\
                        weight_lightvisual*attr_vec['lightvisual']+\
                        weight_familyspiritual*attr_vec['familyspiritual']+\
                        weight_likegirls*attr_vec['likegirls']+\
                        weight_sadness*attr_vec['sadness']+\
                        weight_feelings*attr_vec['feelings']+\
                        weight_danceability*attr_vec['danceability']+\
                        weight_loudness*attr_vec['loudness']+\
                        weight_acousticness*attr_vec['acousticness'] +\
                        weight_instrumental*attr_vec['instrumentalness']+\
                        weight_valence*attr_vec['valence']+\
                        weight_energy*attr_vec['energy']
                
                # update index
                self.index[str(doc_id)] = doc_vec

        elif for_model == 'VSM1_2':
            # obtain vector representations of song attributes
            attr_vec = self.make_vectors()

            for row in tqdm(range(len(self.dataset))):

                # retrieve the id and attribute values for this song
                doc_id = self.dataset.loc[row,'doc_id']
                weight_dating = self.dataset.loc[row,'dating']
                weight_violence = self.dataset.loc[row,'violence']
                weight_worldlife = self.dataset.loc[row,'world/life']
                weight_nighttime = self.dataset.loc[row,'night/time']
                weight_shaketheaudience = self.dataset.loc[row,'shake the audience']
                weight_familygospel = self.dataset.loc[row, 'family/gospel']
                weight_romantic = self.dataset.loc[row, 'romantic']
                weight_communication = self.dataset.loc[row,'communication']
                weight_obscene = self.dataset.loc[row,'obscene']
                weight_music = self.dataset.loc[row,'music']
                weight_movementplaces = self.dataset.loc[row,'movement/places']
                weight_lightvisual = self.dataset.loc[row,'light/visual perceptions']
                weight_familyspiritual = self.dataset.loc[row,'family/spiritual']
                weight_likegirls = self.dataset.loc[row,'like/girls']
                weight_sadness = self.dataset.loc[row, 'sadness']
                weight_feelings = self.dataset.loc[row,'feelings']
                weight_danceability = self.dataset.loc[row,'danceability']
                weight_loudness = self.dataset.loc[row,'loudness']
                weight_acousticness = self.dataset.loc[row, 'acousticness']
                weight_instrumental = self.dataset.loc[row,'instrumentalness']
                weight_valence = self.dataset.loc[row, 'valence']
                weight_energy = self.dataset.loc[row,'energy']

                # calculate the first document vector by summing weighted topic vectors
                doc_vec1 = weight_dating*attr_vec['dating'] + \
                        weight_violence*attr_vec['violence'] + \
                        weight_worldlife*attr_vec['worldlife'] + \
                        weight_nighttime*attr_vec['nighttime'] + \
                        weight_shaketheaudience*attr_vec['shaketheaudience'] +\
                        weight_familygospel*attr_vec['familygospel'] +\
                        weight_romantic*attr_vec['romantic'] +\
                        weight_communication*attr_vec['communication'] +\
                        weight_obscene*attr_vec['obscene'] +\
                        weight_music*attr_vec['music']+\
                        weight_movementplaces*attr_vec['movementplaces']+\
                        weight_lightvisual*attr_vec['lightvisual']+\
                        weight_familyspiritual*attr_vec['familyspiritual']+\
                        weight_likegirls*attr_vec['likegirls']+\
                        weight_sadness*attr_vec['sadness']+\
                        weight_feelings*attr_vec['feelings']
                
                # calculate second document vector by summing weighted (danceability ~ energy) col vectors
                doc_vec2= weight_danceability*attr_vec['danceability']+\
                        weight_loudness*attr_vec['loudness']+\
                        weight_acousticness*attr_vec['acousticness'] +\
                        weight_instrumental*attr_vec['instrumentalness']+\
                        weight_valence*attr_vec['valence']+\
                        weight_energy*attr_vec['energy']
                
                # update index
                self.index[str(doc_id)] = [doc_vec1,doc_vec2]


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
### BM25 ###
BM25_indexer = IndexConstructor('tcc_ceds_music.csv')
print("Initialized Index Constructor for BM25")
print("Constructing index...")

BM25_indexer.construct_index('BM25')
print("Constructed index for BM25")

BM25_indexer.output('BM25_index.txt')
print("BM25 index saved successfully in the format: ", "{'word1':{'doc_list':[(d_i,tf_i),...], 'doc_freq': int}, ...}")
print(len(BM25_indexer.index.keys()), 'words')

### VSM 1.1 ###
VSM_indexer = IndexConstructor('tcc_ceds_music.csv')
print("Initialized Index Constructor for VSM 1_1")
print("Constructing index...")

VSM_indexer.construct_index('VSM1_1')
print("Constructed index for VSM 1_1")

VSM_indexer.output('VSM1_1_index.txt')
print("VSM 1_1 index saved successfully in the format: ", "{'doc_id': doc_vector,...}")
print(len(VSM_indexer.index.keys()), 'documents')

### VSM 1.2 ###
VSM_indexer = IndexConstructor('tcc_ceds_music.csv')
print("Initialized Index Constructor for VSM 1_2")
print("Constructing index...")

VSM_indexer.construct_index('VSM1_2')
print("Constructed index for VSM 1_2")

VSM_indexer.output('VSM1_2_index.txt')
print("VSM 1_2 index saved successfully in the format: ", "{'doc_id': [doc_vector1,doc_vector2], ...}")
print(len(VSM_indexer.index.keys()), 'documents')
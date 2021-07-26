import pickle
import query_preprocessing
import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda


def type_of_nn(query, nn_type=1, method="cosine", n=5):
    music_df = pd.read_csv("tcc_ceds_music.csv")
    
    if nn_type == 1:
        with open('NN_index.txt', 'rb') as handle:
            file=pickle.loads(handle.read())
        

    def nn(query, method, n):
            doc_id_list = list(range(1,28373))
            
            if method == "cosine":
                sim_list = []
                for key, value in tqdm(file.items()):
                    cos_sim = 1 - sp.spatial.distance.cosine(value, query)
                    sim_list.append(cos_sim)
                 #sort from highest value to lowest  
                sorted_ID = [x for _, x in sorted(zip(sim_list, doc_id_list), reverse=True)]
                
            if method == "euclid":
                e_dist_list = []
                for key, value in tqdm(file.items()):
                    dist = np.linalg.norm(value-query)
                    e_dist_list.append(dist)
                #sort from lowest value to highest
                sorted_ID = [x for _, x in sorted(zip(e_dist_list, doc_id_list))]
            
            if method == "dotprod":
                prod_list =[]
                for key, value in tqdm(file.items()):
                    prod = np.inner(value, query)
                    prod_list.append(prod)
                    
                #sort from highest to lowest value
                sorted_ID = [x for _, x in sorted(zip(prod_list, doc_id_list), reverse=True)]
                
            return(sorted_ID[:n], prod_list[:n])   
    
    sorted_ID_final, prod_list_final = nn(query, method, n)            
    recommended_song_infos = []
    for docID in sorted_ID_final[0:n]:
        recommended_song_infos.append(music_df['artist_name'][docID-1] + " - " + music_df['track_name'][docID-1] + ", " + str(music_df['release_date'][docID-1]) + " " + music_df['genre'][docID-1])
                
        
    return(recommended_song_infos,sorted_ID_final[:n], prod_list_final)



# following code only runs if this script is run as main file
if __name__ == '__main__':

    ########## Data Encoding ##########

    # load training dataset
    traindf = pd.read_csv('validation_dataset.csv')


    # assign class index to each song attribute for one-hot encoding; to be used as labels for prediction
    class_labels = {'dating': 0, 'violence': 1, 'world/life': 2, 'night/time':3, 'shake the audience':4, 'family/gospel':5, \
        'romantic':6, 'communication':7, 'obscene':8, 'music':9, 'movement/places':10, 'light/visual perceptions':11, 'family/spiritual':12, \
            'like/girls':13, 'sadness':14, 'feelings':15, 'danceability':16, 'loudness':17, 'acousticness':18, 'instrumentalness':19, 'valence':20, \
                'energy':21}

    # construct new dataframe in format suitable for machine learning
    # only keep queryString and relevant_cols
    traindf = traindf[['queryString', 'relevant_cols']]

    # convert the queryStrings into vector form & convert the relevant cols to one-hot encoding
    # load pre-trained word2vec encoder
    import gensim.downloader as api
    
    # try different word2vec models
    # wv = api.load('word2vec-google-news-300')
    # wv = api.load('glove-wiki-gigaword-300')
    wv = api.load('glove-twitter-200') # note to change line 113, line 159-160

    for row in range(len(traindf)):
        # find column(s) corresponding to this query
        columns = traindf.loc[row,'relevant_cols']
        col_list = columns.split(",")

        # store labels for this query
        classes = []
        
        # class encoding
        for col in col_list:
            classes.append(class_labels[col])
        
        # replace the labels in the train dataframe
        traindf.loc[row,'relevant_cols'] = classes

        # process the query
        query = traindf.loc[row,'queryString']
        tokenQuery = query_preprocessing.process_query(query)
        expandedQuery = query_preprocessing.query_expansion(tokenQuery)

        # vectorise the query
        # q_vec = np.zeros(300)
        q_vec = np.zeros(200)
        for term in expandedQuery:
            if term in wv:
                q_vec += wv[term]
        
        # replace the queries in the train dataframe
        traindf.at[row,'queryString'] = q_vec
        # print(len(q_vec)) # 300

    print(traindf)

    ########## Define PyTorch Dataset ##########

    class MusicDataset(Dataset):
        def __init__(self, queries, targets, transform=None, target_transform=None):
            self.queries = queries
            self.targets = targets
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.queries)

        def __getitem__(self, index):
            # get the query and the corresponding labels for that query
            target = self.targets[index]
            query = self.queries[index]
            # convert to tensors
            if self.transform:
                query = self.transform(query)
            if self.target_transform:
                target = self.target_transform(target)
            # return query and labels
            return query,target

    training_data = MusicDataset(traindf['queryString'], traindf['relevant_cols'], transform=None, \
                    target_transform= Lambda(lambda y: torch.zeros(22, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)))

    train_dataloader = DataLoader(training_data, batch_size=5)

    ########## Build the Neural Network ##########
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            # self.flatten = nn.Flatten()
            self.layer_stack = nn.Sequential(
                # nn.Linear(300, 100),
                nn.Linear(200, 100),
                nn.Sigmoid(),
                nn.Linear(100, 100),
                nn.Sigmoid(),
                nn.Linear(100, 22),
                nn.Softmax()
            )

        def forward(self, x):
            # x = self.flatten(x)
            logits = self.layer_stack(x)
            return logits

    model = NeuralNetwork()

    ########## Define Train loop ##########
    learning_rate = 1e-3
    epochs = 50

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X,y) in enumerate(dataloader):
            # compute prediction and loss
            pred = model(X.float())
            loss = loss_fn(pred, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 2 == 0:
                loss, current = loss.item(), batch*len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
    print("Done!")

    ########## Save Model ##########
    torch.save(model, 'nn_model.pth')

    ########## Load Model ##########
    # model = torch.load('model.pth')

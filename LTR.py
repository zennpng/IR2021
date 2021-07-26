import os.path
import pandas as pd 
import torch
import random

#find gpu otherwise use cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#load and split the data
data = pd.read_csv("LTR_training_data.csv")
data.head()

queries = data['context'].values.tolist()
rel_docs = data['response'].values.tolist()
data = [(queries[i], rel_docs[i]) for i in range(len(queries))]

#Train-Test split
import numpy as np
from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(
    data, test_size=1/10, random_state=179)
train_dataset, valid_dataset = train_test_split(
    train_dataset, test_size=1/9, random_state=179)

#Tokenization
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter

# tokenizer type
tokenizer = get_tokenizer("basic_english")

# vocab
counter = Counter()
for (qry, doc) in train_dataset:
  counter.update(tokenizer(doc))

#define vocab
vocab = torchtext.vocab.Vocab(counter, max_size=10000,  specials=('<pad>', '<unk>'), specials_first=True)
#print("\nVocab size:",len(vocab))


# Dataloader
from torch.utils.data import Dataset, DataLoader

#defined maximum query and doc length
max_doc_len = 1572
max_query_len = 46

#tokenization function
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

#padding function
num_neg_samples = 1

query_padding_pipeline = lambda tokens: [vocab.stoi['<pad>'] for p in range(max_query_len - len(tokens))] + tokens[-max_query_len:]
doc_padding_pipeline = lambda tokens: [vocab.stoi['<pad>'] for p in range(max_doc_len - len(tokens))] + tokens[:max_doc_len]
negative_sampling_pipeline = lambda neg_doc_list: random.sample(neg_doc_list, num_neg_samples)

#collate function for dataloader
train_docs = [d[1] for d in train_dataset]
def collate_batch(batch):

    #initialize empty lists for query and doc lists
    query_list, pos_doc_list, neg_doc_list = [], [], []

    for (qry, doc) in batch:

        #query -> tokens -> id -> pad to max query length
        qry_ = query_padding_pipeline(text_pipeline(qry))

        #doc -> tokens -> ids -> pad to max doc length
        doc_ = doc_padding_pipeline(text_pipeline(doc))

        #negative samples
        irr_doc_list = [d for d in train_docs if d!=doc]
        neg_docs = negative_sampling_pipeline(irr_doc_list)
        neg_docs_ = [doc_padding_pipeline(text_pipeline(d)) for d in neg_docs]

        query_list += [qry_ for q in range(num_neg_samples)]
        pos_doc_list += [doc_ for d in range(num_neg_samples)]
        neg_doc_list += neg_docs_

    #shuffle samples
    temp = list(zip(query_list, pos_doc_list, neg_doc_list))
    random.shuffle(temp)
    query_list, pos_doc_list, neg_doc_list = zip(*temp)

    #Now we have numbers, load them to tensors and put on GPU
    query_list = torch.tensor(query_list, dtype=torch.int64)
    pos_doc_list = torch.tensor(pos_doc_list, dtype=torch.int64)
    neg_doc_list = torch.tensor(neg_doc_list, dtype=torch.int64)
    return query_list.to(device), pos_doc_list.to(device), neg_doc_list.to(device)

BATCH_SIZE=128

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch)


#Model
from torch import nn
class Ranking_model(nn.Module):
    def __init__(self):
        super(Ranking_model, self).__init__()

        #embedding matix, convert tokens to vectors
        self.embedding = nn.Embedding(num_embeddings=len(vocab), 
                                      embedding_dim=50, 
                                      padding_idx=vocab.stoi['<pad>'])

        #defining the LSTM encoder 
        self.encoder = torch.nn.LSTM(input_size=50, hidden_size=50, batch_first=True)

        #feedforward layer
        self.nn_layer1 = nn.Linear(in_features=50*2, out_features=1)


    def forward(self, qry_tokens, pos_doc_tokens, neg_doc_tokens):

        qry_embedded = self.embedding(qry_tokens)
        pos_doc_embedded = self.embedding(pos_doc_tokens)
        neg_doc_embedded = self.embedding(neg_doc_tokens)

        #passing the query, positive, and negative document through the encoder
        out_qry, (enc_qry_vector, ct) = self.encoder(qry_embedded)
        out_pos, (enc_pos_vector, ct) = self.encoder(pos_doc_embedded)
        out_neg, (enc_neg_vector, ct) = self.encoder(neg_doc_embedded)

        #concat query-positive document and query-negative document
        concat_q_pos_doc = torch.cat((out_qry.mean(dim=1), out_pos.mean(dim=1)), dim=1)
        concat_q_neg_doc = torch.cat((out_qry.mean(dim=1), out_neg.mean(dim=1)), dim=1)

        #feed to linear layer
        pos_score = torch.relu(self.nn_layer1(concat_q_pos_doc))
        neg_score = torch.relu(self.nn_layer1(concat_q_neg_doc))

        diff = pos_score - neg_score
        return diff

# Construct our model by instantiating the model class defined above
model = Ranking_model()
model.to(device) #put it on the device

from gensim.models import KeyedVectors
import gensim.downloader as api

try:
    print("Loading saved word vectors...")
    glove_50dim = KeyedVectors.load("./LTR_glove_50dim.w2v")
except:
    print("Downloading word vectors...")
    glove_50dim = api.load("glove-wiki-gigaword-50")
    glove_50dim.save('LTR_glove_50dim.w2v')

print("Number of word vectors:", glove_50dim.vectors.shape)

#Initialise model embedding with glove
for word in vocab.stoi.keys():
    if word in glove_50dim.key_to_index.keys():
        word_vec = glove_50dim[word]
        model.embedding.weight.data[vocab.stoi[word]] = torch.tensor(word_vec)

#function to score documents based on a query
#to be used for a trained model
def rank_docs(qry, doc_list):
    for doc in doc_list:
        model.eval()
        with torch.no_grad():
            qry_ = torch.tensor([query_padding_pipeline(text_pipeline(qry))], dtype=torch.int64).to(device)
            doc_ = torch.tensor([doc_padding_pipeline(text_pipeline(doc))], dtype=torch.int64).to(device)
            score = model(qry_, doc_, doc_*0)
            print(score)
            print("query [{}] to doc [{}] matching score [{}]\n".format(qry, doc, score.detach().item()))


if not os.path.exists("LTR_model.pth"):  # If model does not exist, train and save model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) #optimizer
    num_epochs = 10 #epochs

    for epoch in range(num_epochs):
        print("-->Epoch:{}".format(epoch))

        epoch_train_loss = 0.0
        model.train()
        for idx, (qry_tokens, pos_doc_tokens, neg_doc_tokens) in enumerate(train_dataloader):

            #flush the gradient values
            optimizer.zero_grad()

            #calculate model output
            diff = model(qry_tokens, pos_doc_tokens, neg_doc_tokens)

            #pairwise loss 
            loss = torch.mean(torch.log(1+torch.exp(-1.0*diff)))

            #backward pass
            loss.backward() 

            #weights update
            optimizer.step()

            #average train loss
            epoch_train_loss += loss.cpu().item()*BATCH_SIZE

            print("Batch {}/{}, avg. train loss is {}".format(idx, len(train_dataloader), epoch_train_loss/(idx+1)), end='\r')

    torch.save(model, "LTR_model.pth")


else:  # else, validate (do ranking)
    model = torch.load("LTR_model.pth")

    epoch_val_loss = 0.0
    model.eval()
    with torch.no_grad():  # weights should not update
        for idx, (qry_tokens, pos_doc_tokens, neg_doc_tokens) in enumerate(valid_dataloader):

            #formward pass
            diff = model(qry_tokens, pos_doc_tokens, neg_doc_tokens) 

            epoch_val_loss += torch.mean(torch.log(1+torch.exp(-1.0*diff)))  # same loss as in training

        #print("\nval loss:{}".format(epoch_val_loss))

        qry = "we study at the university"
        doc1 = "the school has a good student to teacher ratio"
        doc2 = "we have has many students"
        doc3 = "singapore has a decent climate"

        rank_docs(qry, [doc1, doc2, doc3])


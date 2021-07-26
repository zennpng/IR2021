import nltk
from itertools import chain
from more_itertools import locate
from nltk.corpus import wordnet as wn
from numpy import exp2

def process_query(query):

    # define stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.extend(['please', 'want', 'like', 'music', 'thanks', 'need', 'require', 'feeling'])  # extend stopwords manually
    
    # tokenize query
    query_tokens = nltk.word_tokenize(query)

    # replace "not something" with the antonym of "something"
    not_loc = list(locate(query_tokens, lambda x: x == 'not'))
    targets = [query_tokens[i+1] for i in not_loc]
    for index in sorted(not_loc, reverse=True):
        del query_tokens[index+1]
        del query_tokens[index]
    expansion = PyDictionary(targets).getAntonyms()
    for target in expansion:
        query_tokens.append(target[next(iter(target))][0])

    # remove stopwords
    query_tokens = [w for w in query_tokens if w.lower() not in stopwords]

    return query_tokens



def query_expansion_old(query_list):  ### DEPRECATED (DOES NOT LOOK VERY ACCURATE)
    expanded_query = []
    for word in query_list:
        synonyms = wn.synsets(word)
        expansion = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        expansion = [ x for x in expansion if "_" not in x ]
        expanded_query += expansion
    return(expanded_query)



# NEW QUERY EXPANSION (more conservative)
import pickle
from PyDictionary import PyDictionary

# get global lyrics pool of words
with open("lyrics_vocab.txt", "rb") as fp: 
    lyrics_vocab = pickle.load(fp)

def query_expansion(query_list):

    # get word synonyms
    expansion = PyDictionary(query_list).getSynonyms()
    for word_expanded in expansion:
        if word_expanded != None:
            query_list += word_expanded[next(iter(word_expanded))]

    # remove phrases and words which are not part of the global lyrics pool of words
    query_list = [ x for x in query_list if " " not in x and x in lyrics_vocab ]
    
    return query_list
        
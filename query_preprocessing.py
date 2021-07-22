import nltk
from itertools import chain
from nltk.corpus import wordnet as wn
from numpy import exp2

def process_query(query):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.extend(['please', 'want', 'like', 'music', 'thanks', 'need', 'require'])  # extend stopwords manually
    query_tokens = nltk.word_tokenize(query)
    query_tokens = [w for w in query_tokens if w.lower() not in stopwords]
    return query_tokens

# test_query = "i need some workout energy now please thanks"
# print(process_query(test_query))
# ['workout', 'energy']

def query_expansion_old(query_list):
    expanded_query = []
    for word in query_list:
        synonyms = wn.synsets(word)
        expansion = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        expansion = [ x for x in expansion if "_" not in x ]
        expanded_query += expansion
    return(expanded_query)

#print(query_expansion_old(["gym","tired","motivation","exertion"]))


# NEW QUERY EXPANSION (more conservative)
import pickle
from PyDictionary import PyDictionary

with open("lyrics_vocab.txt", "rb") as fp: 
    lyrics_vocab = pickle.load(fp)

def query_expansion(query_list):
    expansion = PyDictionary(query_list).getSynonyms()
    for word_expanded in expansion:
        query_list += word_expanded[next(iter(word_expanded))]
    query_list = [ x for x in query_list if " " not in x and x in lyrics_vocab ]
    return query_list
        
print(query_expansion(["energy", "workout", "feelings"]))
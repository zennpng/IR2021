import nltk
from itertools import chain
from nltk.corpus import wordnet as wn

def process_query(query):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.extend(['please', 'want', 'like', 'music', 'thanks', 'need', 'require'])  # extend stopwords manually
    query_tokens = nltk.word_tokenize(query)
    query_tokens = [w for w in query_tokens if w.lower() not in stopwords]
    return query_tokens

# test_query = "i need some workout energy now please thanks"
# print(process_query(test_query))
# ['workout', 'energy']

def query_expansion(query_list):
    expanded_query = []
    for word in query_list:
        synonyms = wn.synsets(word)
        expansion = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        expansion = [ x for x in expansion if "_" not in x ]
        expanded_query += expansion
    return(expanded_query)

#print(query_expansion(["gym","tired","motivation","exertion"]))

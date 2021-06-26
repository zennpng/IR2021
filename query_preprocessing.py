import nltk

def process_query(query):
    stopwords = nltk.corpus.stopwords.words("english")
    query_tokens = nltk.word_tokenize(query)
    query_tokens = [w for w in query_tokens if w.lower() not in stopwords]
    return query_tokens

test_query = "i woke up and choose violence"
# print(process_query(test_query))
# ['need', 'workout', 'energy']



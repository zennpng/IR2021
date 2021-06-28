import nltk

def process_query(query):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.extend(['please', 'want', 'like', 'music', 'thanks', 'need', 'require'])  # extend stopwords manually
    query_tokens = nltk.word_tokenize(query)
    query_tokens = [w for w in query_tokens if w.lower() not in stopwords]
    return query_tokens

# test_query = "i need some workout energy now please thanks"
# print(process_query(test_query))
# ['workout', 'energy']



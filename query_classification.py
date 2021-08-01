import math
import random
import numpy as np
import dataset_preprocessing
import LangModel_preprocessing

# used for unseen words in training vocabularies
UNK = None

class UnigramLanguageModel:
    def __init__(self, sentences, mode="collection", smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                self.corpus_length += 1
        self.unique_words = len(self.unigram_frequencies)
        self.smoothing = smoothing
        self.mode = mode

    def calculate_unigram_probability(self, word):
            word_probability_numerator = self.unigram_frequencies.get(word, 0)
            word_probability_denominator = self.corpus_length
            if self.smoothing:
                word_probability_numerator += 1
                if self.mode == "collection":
                    # add one more to total number of seen unique words for UNK - unseen events
                    word_probability_denominator += self.unique_words + 1
                else:
                    word_probability_denominator += indexed_terms
            return float(word_probability_numerator) / float(word_probability_denominator)

    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        sentence_probability_log_sum = 0
        for word in sentence:
            word_probability = self.calculate_unigram_probability(word)
            sentence_probability_log_sum += math.log(word_probability, 2)
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum                

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.sort()
        full_vocab.append(UNK)
        return full_vocab

# calculate number of unigrams & bigrams
def calculate_number_of_unigrams(sentences):
    unigram_count = 0
    for sentence in sentences:
        unigram_count += len(sentence)
    return unigram_count

# calculate interpolated sentence/query probability
def calculate_interpolated_sentence_probability(sentence, doc, collection, alpha=0.75, normalize_probability=True):
        sentence_probability_log_sum = 0
        for word in sentence:
            word_probability1 = doc.calculate_unigram_probability(word)
            word_probability2 = collection.calculate_unigram_probability(word)
            word_probability = alpha*word_probability1 + (1-alpha)*word_probability2
            sentence_probability_log_sum += math.log(word_probability, 2)
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum 

# print unigram and bigram probs
def print_unigram_probs(sorted_vocab_keys, model):
    for vocab_key in sorted_vocab_keys:
        print("{}: {}".format(vocab_key if vocab_key != UNK else "UNK",
                                    model.calculate_unigram_probability(vocab_key)), end=" ")
    print("")

# topic classification splitting
data_topicsplit = []
df = dataset_preprocessing.musicdf
topics = df.topic.unique()
for topic in topics:
    lyrics_topic = df[df['topic'] == topic]["lyrics"].tolist()
    #lyrics_topic = random.sample(lyrics_topic, 100)
    #print(lyrics_topic[0:2])
    data_topicsplit.append(lyrics_topic)

def query_langmodel(query):
    actual_dataset = LangModel_preprocessing.lyrics_list
    actual_dataset_model_smoothed = UnigramLanguageModel(actual_dataset, mode = "collection", smoothing=True)
 
    global indexed_terms
    indexed_terms = actual_dataset_model_smoothed.unique_words + 1 # add 1 for UNK
    
    score_list = []
    for lyric in data_topicsplit:
        lyric_dataset_model_smoothed = UnigramLanguageModel([lyric], mode="doc", smoothing=True)
        score = calculate_interpolated_sentence_probability(query, lyric_dataset_model_smoothed, actual_dataset_model_smoothed)
        score_list.append(score)
    score_list = [float(i)/sum(score_list) for i in score_list]
    score_array = np.array(score_list)
    selected_topicID = score_array.argsort()[-1:][::-1][0]
    return score_array, topics[selected_topicID]

test_queries = [["feelings", "emotions"],
                ["hate", "kill"],
                ["love", "lovely"]]

for test_query in test_queries:
    print(query_langmodel(test_query))

## It appears that this does not work; Every query gets categorized under the "feelings" category, reasons are unknown
## When the same number of songs are sampled from each of the categories, the scores for each topic become the same, reasons are unknown
'''
sadness       6096
violence      5710
world/life    5420
obscene       4882
music         2303
night/time    1825
romantic      1524
feelings       612
'''
import re
import math
from tqdm import tqdm
import numpy as np
import LangModel_preprocessing
from dataset_preprocessing import musicdf

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

def langmodel(query, n=5):
    actual_dataset = LangModel_preprocessing.lyrics_list
    actual_dataset_model_smoothed = UnigramLanguageModel(actual_dataset, mode = "collection", smoothing=True)

    global indexed_terms
    indexed_terms = actual_dataset_model_smoothed.unique_words + 1 # add 1 for UNK
    
    score_list = []
    for lyric in tqdm(actual_dataset):
        lyric_dataset_model_smoothed = UnigramLanguageModel([lyric], mode="doc", smoothing=True)
        score = calculate_interpolated_sentence_probability(query, lyric_dataset_model_smoothed, actual_dataset_model_smoothed)
        score_list.append(score)

    score_array = np.array(score_list)
    selected_docsID = list(score_array.argsort()[-n:][::-1])

    recommended_song_infos = []
    for docID in selected_docsID:
        recommended_song_infos.append(musicdf['artist_name'][docID-1] + " - " + musicdf['track_name'][docID-1] + ", " + str(musicdf['release_date'][docID-1]) + " " + musicdf['genre'][docID-1])
    return selected_docsID, recommended_song_infos


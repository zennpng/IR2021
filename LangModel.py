import re
import math
import numpy as np

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]

class UnigramLanguageModel:
    def __init__(self, sentences, mode="collection", smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
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
            if word != SENTENCE_START and word != SENTENCE_END:
                word_probability = self.calculate_unigram_probability(word)
                sentence_probability_log_sum += math.log(word_probability, 2)
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum                

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.remove(SENTENCE_START)
        full_vocab.remove(SENTENCE_END)
        full_vocab.sort()
        full_vocab.append(UNK)
        full_vocab.append(SENTENCE_START)
        full_vocab.append(SENTENCE_END)
        return full_vocab

# calculate number of unigrams & bigrams
def calculate_number_of_unigrams(sentences):
    unigram_count = 0
    for sentence in sentences:
        # remove two for <s> and </s>
        unigram_count += len(sentence) - 2
    return unigram_count

# calculate interpolated sentence/query probability
def calculate_interpolated_sentence_probability(sentence, doc, collection, alpha=0.75, normalize_probability=True):
        sentence_probability_log_sum = 0
        for word in sentence:
            if word != SENTENCE_START and word != SENTENCE_END:
                word_probability1 = doc.calculate_unigram_probability(word)
                word_probability2 = collection.calculate_unigram_probability(word)
                #print(word_probability1)
                #print(word_probability2)
                word_probability = alpha*word_probability1 + (1-alpha)*word_probability2
                #print(word_probability2)
                sentence_probability_log_sum += math.log(word_probability, 2)
                #print(sentence_probability_log_sum)
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum 

# print unigram and bigram probs
def print_unigram_probs(sorted_vocab_keys, model):
    for vocab_key in sorted_vocab_keys:
        if vocab_key != SENTENCE_START and vocab_key != SENTENCE_END:
            print("{}: {}".format(vocab_key if vocab_key != UNK else "UNK",
                                       model.calculate_unigram_probability(vocab_key)), end=" ")
    print("")

if __name__ == '__main__':
    actual_dataset = read_sentences_from_file("./train.txt")
    doc1_dataset = read_sentences_from_file("./doc1.txt")
    doc2_dataset = read_sentences_from_file("./doc2.txt")
    doc3_dataset = read_sentences_from_file("./doc3.txt")
    actual_dataset_test = read_sentences_from_file("./test.txt")

    actual_dataset_model_smoothed = UnigramLanguageModel(actual_dataset, mode = "collection", smoothing=True)

    global indexed_terms
    indexed_terms = actual_dataset_model_smoothed.unique_words + 1 # add 1 for UNK
    
    doc1_dataset_model_smoothed = UnigramLanguageModel(doc1_dataset, mode="doc", smoothing=True)
    doc2_dataset_model_smoothed = UnigramLanguageModel(doc2_dataset, mode="doc", smoothing=True)
    doc3_dataset_model_smoothed = UnigramLanguageModel(doc3_dataset, mode="doc", smoothing=True)

    #print(str(actual_dataset_model_smoothed.unique_words))
    #sorted_vocab_keys = actual_dataset_model_smoothed.sorted_vocabulary()
    #print_unigram_probs(sorted_vocab_keys, actual_dataset_model_smoothed)
    print("== QUERY PROBABILITIES == ")
    longest_sentence_len = max([len(" ".join(sentence)) for sentence in actual_dataset_test])
    for sentence in actual_dataset_test:
        sentence_string = " ".join(sentence)
        print(sentence_string)
        score_query_doc1 = calculate_interpolated_sentence_probability(sentence, doc1_dataset_model_smoothed, actual_dataset_model_smoothed)
        score_query_doc2 = calculate_interpolated_sentence_probability(sentence, doc2_dataset_model_smoothed, actual_dataset_model_smoothed)
        score_query_doc3 = calculate_interpolated_sentence_probability(sentence, doc3_dataset_model_smoothed, actual_dataset_model_smoothed)

        print(score_query_doc1)
        print(score_query_doc2)
        print(score_query_doc3)

        best_doc = np.argmax([score_query_doc1, score_query_doc2, score_query_doc3])

        print("Best matched doc is: doc" + str(best_doc+1))

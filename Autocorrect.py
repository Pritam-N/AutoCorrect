import re
from re import match
import numpy as np
import pandas as pd
from collections import OrderedDict

from torch import cos
from Levenshtein import *
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util

vocab_file = 'Vocab/tokens.txt'

def process_vocab(vocab_file='Vocab/tokens.txt'):
    words = []

    with open(vocab_file, encoding="utf8") as f:
        line = f.read()
        line_lower = line.lower()
        words = re.findall(r'\w+', line_lower)

    return words

class AutoCorrect():
    def __init__(self, vocab_file='Vocab/tokens.txt'):
        self.vocab_file = vocab_file
        self.words = set(process_vocab(vocab_file))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    def get_counts(self):

        word_count_dict = {}
        for w in self.words:
            word_count_dict[w] = word_count_dict.get(w, 0) + 1

        return word_count_dict

    def word_probs(self, word_count_dict):
        probs = {}
        length = sum(word_count_dict.values())
        for k in word_count_dict:
            probs[k] = word_count_dict[k] / length
        return probs

    def delete_letter(self, word):
        delete_l = []
        split_l = []
        split_l = [(word[:i],word[i:]) for i in range(len(word) + 1)]
        delete_l = [L+R[1:] for L,R in split_l if R]
        return delete_l

    def switch_letter(self, word):
        switch_l = []
        split_l = []
        split_l = [(word[:i],word[i:]) for i in range(len(word) + 1)]
        switch_l = [L + R[1] + R[0] + R[2:] for L,R in split_l if len(R) > 1]

        return switch_l

    def replace_letter(self, word):
        replace_l = []
        split_l = []
        letters = 'abcdefghijklmnopqrstuvwxyz- '
        split_l = [(word[:i],word[i:]) for i in range(len(word) + 1)]
        replace_l = [L + alphabet + R[1:] for L,R in split_l if R for alphabet in letters]
        replace_l.remove(word)

        return replace_l

    def insert_letter(self, word):
        insert_l = []
        split_l = []
        letters = 'abcdefghijklmnopqrstuvwxyz- '
        split_l = [(word[:i],word[i:]) for i in range(len(word) + 1)]
        insert_l = [L + alphabet + R for L,R in split_l if R for alphabet in letters]

        return insert_l

    def edit_one_letter(self, word):
        updates = self.delete_letter(word) + \
                    self.switch_letter(word) + \
                    self.replace_letter(word) + \
                    self.insert_letter(word)

        updates = set(updates)
        updates.discard(word)
        return updates

    def run_recursive(self, list_elems):
        updates = []
        for value in list_elems:
            updates = updates + list(self.edit_one_letter(value))
        return updates

    def running_predictions(self, word, updates):
        
        matching = []
        for value in updates:
            if value in self.words:
                matching.append(value)

        dict_matching = {}
        for value in matching:
            dict_matching[value] = Levenshtein.claculate_distance(word, value)

        dict_updates = dict(sorted(dict_matching.items(), key=lambda item: item[1]))
        return dict_updates, matching

    def find_words(self, word):
        updates = list(self.edit_one_letter(word))

        dict_updates, matching = self.running_predictions(word, updates)

        if not matching:
            updates = self.run_recursive(updates)
            updates.remove(word)
            dict_updates, matching = self.running_predictions(word, updates)

        return dict_updates

    def get_optimal_letter(self, word):
        dict_updates = self.find_words(word)
        dict_updates = sorted(dict_updates.items(), key=lambda item: item[1])
        top_5 = []
        for i in dict_updates[:5]:
            (k,v) = i
            top_5.append(k)
        return top_5

    def prepare_sentence(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens_list = []
        for i in range(len(tokens)):
            if i> 0 and tokens[i].startswith('##'):
                if tokens[i-1] in tokens_list:
                    tokens_list.remove(tokens[i-1])
                tokens_list.append(tokens[i-1] + tokens[i][2:])
            else:
                tokens_list.append(tokens[i])

        out_of_vocab = []
        for i in range(len(tokens_list)):
            word = tokens_list[i]
            if word not in self.words:
                out_of_vocab.append(word)

        return out_of_vocab

    def replace_out_of_vocab(self, out_of_vocab, sentence):
        new_sentences = []
        word_obj = {}
        corpus_embedding = self.model.encode(sentence, convert_to_tensor=True)
        for word in out_of_vocab:
            top_5 = self.get_optimal_letter(word)
            if len(top_5) > 0:
                similarity = 0.0
                correct_word = top_5[0]
                for new_word in top_5:
                    #new_sentences.append(sentence.replace(word, new_word))
                    new_sentence = sentence.replace(word, new_word)
                    query_embedding = self.model.encode(new_sentence, convert_to_tensor=True)
                    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embedding)[0]
                    cos_scores = cos_scores.cpu()
                    if cos_scores > similarity:
                        similarity = cos_scores
                        correct_word = new_word
                word_obj[word] = correct_word

        return word_obj

    def process_sentence(self, sentence):
        out_of_vocab = self.prepare_sentence(sentence)

        words = self.replace_out_of_vocab(out_of_vocab, sentence)

        return words               

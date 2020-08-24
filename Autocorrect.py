import re
from re import match
import numpy as np
import pandas as pd
from collections import OrderedDict
from Levenshtein import *

vocab_file = 'Vocab/vocab.txt'

def process_vocab(vocab_file='Vocab/vocab.txt'):
    words = []

    with open(vocab_file, encoding="utf8") as f:
        line = f.read()
        line_lower = line.lower()
        words = re.findall(r'\w+', line_lower)

    return words

class AutoCorrect():
    def __init__(self, vocab_file='Vocab/vocab.txt'):
        self.vocab_file = vocab_file
        self.words = set(process_vocab(vocab_file))

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
        letters = 'abcdefghijklmnopqrstuvwxyz'
        split_l = [(word[:i],word[i:]) for i in range(len(word) + 1)]
        replace_l = [L + alphabet + R[1:] for L,R in split_l if R for alphabet in letters]
        replace_l.remove(word)

        return replace_l

    def insert_letter(self, word):
        insert_l = []
        split_l = []
        letters = 'abcdefghijklmnopqrstuvwxyz'
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

        dict_updates = dict(sorted(dict_matching.items()))
        return dict_updates, matching

    def find_words(self, word):
        updates = list(self.edit_one_letter(word))

        dict_updates, matching = self.running_predictions(word, updates)

        if not matching:
        
            while True:
                updates = self.run_recursive(updates)
                updates.remove(word)
                dict_updates, matching = self.running_predictions(word, updates)
                if matching:
                    break

        return dict_updates

    def get_optimal_letter(self, word):
        dict_updates = self.find_words(word)
        top_5 = []
        for _ in range(1,6):
            top_5.append(next(iter(dict_updates)))
        return top_5
from logging import error
from os import replace
import numpy as np
import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForMaskedLM

class MaskPrediction():
    def __init__(self) -> None:
        self.bert_model = 'bert-large-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.model = BertForMaskedLM.from_pretrained(self.bert_model)
        self.model.eval()

    def get_bert_candidates(self, input_text, error_words, max_predictions=10):
        list_candidates_bert = []
        if error_words:
            input_text_split = input_text.split(' ')
            if len(input_text_split) > 3:
                for word, error_word in zip(input_text.split(), error_words):
                    if error_word:
                        replace_word_mask = input_text.replace(word, '[MASK]')
                        text = f'[CLS]{replace_word_mask} [SEP] {input_text} [SEP] '
                        tokenize_text = self.tokenizer.tokenize(text)
                        masked_index = [i for i, x in enumerate(tokenize_text) if x == '[MASK]'][0]
                        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenize_text)
                        segment_ids = [0]*len(tokenize_text)
                        token_tensor = torch.tensor([indexed_tokens])
                        segment_tensor = torch.tensor([segment_ids])
                        with torch.no_grad():
                            outputs = self.model(token_tensor, token_type_ids = segment_tensor)
                            predictions = outputs[0][0][masked_index]
                        predicted_ids = torch.argsort(predictions, descending=True)[:max_predictions]
                        predicted_tokens = self.tokenizer.convert_ids_to_tokens(list(predicted_ids))
                        list_candidates_bert.append((word, predicted_tokens))
                return list_candidates_bert

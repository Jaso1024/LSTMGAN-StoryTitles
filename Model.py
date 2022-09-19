import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten
from tensorflow_text import BertTokenizer
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer

import numpy as np
import os


class Encoder():
    def __init__(self, text) -> None:
        self.vocab = self.get_vocab(text)
        lookup_table = StaticVocabularyTable(
            KeyValueTensorInitializer(
                keys=self.vocab,
                key_dtype=tf.string,
                values=tf.range(1, len(self.vocab)+1, dtype=tf.int64)
            ), 
            num_oov_buckets=100
        )
        self.tokenizer = BertTokenizer(lookup_table)

    def get_vocab(self, text):
        text = text.split(" ")
        return list(set(text)) 
    
    def get_vocab_length(self):
        return len(self.vocab)
    
    def encode(self, text):
        return self.tokenizer.tokenizer(text)

    def decode(self, tokens):
        return self.tokenizer.detokenize(tokens)    


    

def test_encoder():
    text = "Hello my name is BERT I am an AI used for NLP"
    encoder = Encoder(text)
    tokens = encoder.tokenizer.tokenize("I am BERT")
    print(tokens)
    print(encoder.tokenizer.detokenize(tokens))



    

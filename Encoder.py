import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten, Reshape
from tensorflow_text import BertTokenizer, WordpieceTokenizer
from keras.utils import pad_sequences
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer
import numpy as np
import os
from itertools import repeat
import time

class Encoder():
    def __init__(self, text, maxlen) -> None:
        self.vocab = self.get_vocab(text)
        self.vocab_len = len(self.vocab)
        lookup_table = StaticVocabularyTable(
            KeyValueTensorInitializer(
                keys=self.vocab,
                key_dtype=tf.string,
                values=tf.range(0, self.vocab_len, dtype=tf.int64)
            ), 
            num_oov_buckets=100
        )
        self.tokenizer = BertTokenizer(lookup_table)

        self.sequence_maxlen = maxlen
        self.identity = np.identity(self.vocab_len)

    def get_vocab(self, text):
        vocab = set()
        for title in text:
            for word in title.split(" "):
                vocab.add(word)

        return [" "]+list(vocab) 
    

    def get_maxlen(self, text):
        maxlen = 0
        for title in text:
            current_length = len(self.encode_no_padding(title))
            maxlen = current_length if current_length > maxlen else maxlen

        
        return maxlen

    def pad(self, sequence):
        return pad_sequences(sequence, maxlen=self.sequence_maxlen, padding="post")

    def encode_no_padding(self, text):
        return [token for token in self.tokenizer.tokenize(text)[0]]

    def encode(self, text):
        tokens = [token[0].numpy() for token in self.tokenizer.tokenize(text)[0]]
        tokens = np.array(tokens, dtype=np.float32)
        tokens = self.pad([tokens])
        tokens = [self.identity[token] for token in tokens[0]]
        tokens = tf.expand_dims(tokens, axis=0)
        tokens = np.array(tokens, dtype=np.float32) # Must do this again to prevent tensorflow from removing 0s
        
        return tokens

    def decode(self, tokens):
        tensor = self.tokenizer.detokenize([tokens])
        print(tf.get_static_value(tensor[0][0]))
        return " ".join([tf.get_static_value(word)[0].decode("utf-8") for word in tensor[0]])
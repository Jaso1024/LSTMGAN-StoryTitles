

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten, Reshape, Conv2D, Dropout, InputLayer, LeakyReLU
from tensorflow_text import BertTokenizer, WordpieceTokenizer
from keras.utils import pad_sequences
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer
import numpy as np
import os
from itertools import repeat
import time
import pandas as pd
from nltk.corpus import words



class AutoEncoder(Model):
    def __init__(self, vocab_len, sequence_len, encoding_size=32):
        super(AutoEncoder, self).__init__()

        self.vocab_len = vocab_len

        self.i = InputLayer((1, sequence_len, vocab_len))
        self.e1 = GRU(128, activation="sigmoid", return_sequences=True)
        self.e2 = Dropout(0.2)
        self.e3 = GRU(64, activation="sigmoid", return_sequences=True)
        self.e4 = Dropout(0.2)
        self.e5 = GRU(encoding_size, activation="sigmoid", return_sequences=False)
        

        
        self.d7 = Dense(vocab_len, activation="softmax")
        self.d6 = Dropout(0.2)
        self.d5 = GRU(64, activation="sigmoid", return_sequences=True)
        self.d4 = Dropout(0.2)
        self.d3 = GRU(encoding_size, activation="sigmoid", return_sequences=True)
        self.d2 = Reshape((sequence_len, vocab_len))
        self.d1 = Dense(sequence_len*vocab_len, activation="relu")

        

    def call(self, inp):
        x = self.encode(inp)
        x = self.decode(x)
        return x

    def encode(self, inp):
        x = self.i(inp)
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        

        return x
    
    def decode(self, inp):
        x = self.d1(inp)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.d7(x)

        return x
    



    
        
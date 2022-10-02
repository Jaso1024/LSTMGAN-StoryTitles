
import tensorflow as tf
from tensorflow import keras
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
import tensorflow_probability as tfp
from nltk.corpus import words



class AutoEncoder(Model):
    def __init__(self, vocab_len, sequence_len, encoding_size=128):
        super(AutoEncoder, self).__init__()

        self.vocab_len = vocab_len

        self.i = InputLayer((1, sequence_len, vocab_len))
        self.e1 = Flatten()
        self.e2 = Dense(512, activation=None)
        self.e3 = LeakyReLU(0.2)
        self.e4 = Dense(256, activation=None)
        self.e5 = LeakyReLU(0.2)
        self.e6 = Dense(encoding_size, activation="tanh")
        

        self.d10 = Dense(vocab_len, activation="softmax")
        self.d9 = LeakyReLU(0.2)
        self.d8 = Dense(256, activation=None)
        self.d7 = LeakyReLU(0.2)
        self.d6 = Dense(256, activation=None)
        self.d5 = Reshape((sequence_len, vocab_len))
        self.d4 = LeakyReLU(0.2)
        self.d3 = Dense(sequence_len*vocab_len, activation=None)
        self.d2 = LeakyReLU(0.2)
        self.d1 = Dense(encoding_size, activation=None)

        

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
        x = self.e6(x)

        return x
    
    def decode(self, inp):
        x = self.d1(inp)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.d7(x)
        x = self.d8(x)
        x = self.d9(x)
        x = self.d10(x)

        return x
    
if __name__ == "__main__":
    e = AutoEncoder()
    data = pd.read_pickle("nosleep_data.pkl")
    categories = data.category.tolist()
    titles = data.title.tolist()
    formatted_titles = []
    words = set(words.words())
    for title in titles:
        title = title.lower()
        title = title.replace(".", "")
        title = title.replace("?", "")
        title = title.replace(")", "")
        if len(title.split(" "))>25:
            continue
        elif len(title.split(" ")) < 3:
            continue
        elif not all(word in words for word in title.split(" ")):
            continue
            
        formatted_titles.append(title)
    


    
        
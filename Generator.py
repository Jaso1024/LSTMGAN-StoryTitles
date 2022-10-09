from pydoc import describe
from typing import Container
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten, Reshape, Conv2D, Dropout, LeakyReLU
from tensorflow_text import BertTokenizer, WordpieceTokenizer
from keras.utils import pad_sequences
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer
import numpy as np
import os
from itertools import repeat
import time


class Generator(Model):
    def __init__(self, len_vocab, sequence_len):
        super(Generator, self).__init__()

        self.vocab_len = len_vocab
        self.sequence_len = sequence_len

        self.flat = Flatten()
        self.concat = Concatenate()
        

        self.l1 = GRU(256, activation="sigmoid", return_sequences=True)
        self.l2 = GRU(64, activation="sigmoid",return_sequences=True)
        self.l3 = GRU(sequence_len, activation="sigmoid",return_sequences=False)


    def call(self, noise):
        x = self.l1(noise)
        x = self.l2(x)
        x = self.l3(x)
        
        return x
    

    
    def get_padding(self, iteration):
        num_zeros = self.sequence_len - iteration
        return [0] * num_zeros



if __name__ == "__main__":

    generator = Generator(5, 5)
    generator.compile(optimizer="adam", loss="mse")
    noise = np.random.randint(low=0, high=1000, size=(1,1,1000))
    print(generator.predict(noise[0]))
    ys = np.random.randint(low=1, high=5, size=(1,5))
    generator.fit(noise,np.array(ys, dtype=np.int32))
    
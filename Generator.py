from typing import Container
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten, Reshape, Conv2D, Dropout
from tensorflow_text import BertTokenizer, WordpieceTokenizer
from keras.activations import leaky_relu
from keras.utils import pad_sequences
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer
import numpy as np
import os
from itertools import repeat
import time
import tensorflow_probability as tfp


class Generator(Model):
    def __init__(self, len_vocab, sequence_len):
        super(Generator, self).__init__()

        self.vocab_len = len_vocab
        self.sequence_len = sequence_len

        self.flat = Flatten()
        self.concat = Concatenate()
        

        self.n1 = Dense(512, activation="relu")
        self.n2 = Dense(256, activation="relu")

        self.t1 = GRU(512, input_shape=(1, self.sequence_len, self.vocab_len), activation="sigmoid", return_sequences=True)
        self.t2 = Dropout(0.2)
        self.t3 = GRU(512, input_shape=(1, self.sequence_len, self.vocab_len), activation="sigmoid")
        self.t4 = Dropout(0.2)

        self.fc1 = Dense(512, activation="elu")
        self.fc2 = Dense(512, activation="elu")
        self.out = Dense(self.vocab_len, activation="softmax")

    def call(self, noise, tokens):
        n = self.n1(noise)
        n = self.n2(n)
        n = self.flat(n)

        t = self.t1(tokens)
        t = self.t2(t)
        t = self.t3(t)
        t = self.t4(t)
        t = self.flat(t)

        x = self.concat([n, t])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        
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
    
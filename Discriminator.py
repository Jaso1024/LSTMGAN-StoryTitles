import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten, Reshape, LeakyReLU, BatchNormalization, Dropout
from tensorflow_text import BertTokenizer, WordpieceTokenizer
from keras.utils import pad_sequences
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer
import numpy as np
import os
from itertools import repeat
import time

from Generator import Generator


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.flat = Flatten()

        self.l1 = Dense(256, activation=None)
        self.l2 = LeakyReLU(0.2)
        self.d1 = Dropout(0.2)
        self.l3 = Dense(256, activation=None)
        self.l4 = LeakyReLU(0.2)
        self.d2 = Dropout(0.2)
        self.l5 = Dense(256, activation=None)
        self.l6 = LeakyReLU(0.2)
        self.d3 = Dropout(0.2)
        self.l7 = Dense(1, activation="sigmoid")

    def call(self, text):
        x = self.l1(text)
        x = self.l2(x)
        x = self.d1(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.d2(x)
        x = self.flat(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.d3(x)
        x = self.l7(x)

        return x

if __name__ == "__main__":
    generator = Generator(5, 5)
    generator.compile(optimizer="adam", loss="mse")
    noise = np.random.randint(low=0, high=1000, size=(1,1,1000))
    text = generator.predict(noise[0])
    text = np.reshape(text, (5,5))
    text = tf.expand_dims(text, axis=0)
    discriminator = Discriminator()
    discriminator.compile(optimizer="adam", loss="mse")
    discriminator.predict(text)

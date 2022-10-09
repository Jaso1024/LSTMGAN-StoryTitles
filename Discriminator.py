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

        self.l1 = GRU(256, activation="sigmoid", return_sequences=True)
        self.d1 = Dropout(0.2)
        self.l2 = GRU(128, activation="sigmoid",return_sequences=True)
        self.d2 = Dropout(0.2)
        self.l3 = GRU(64, activation="relu",return_sequences=False)
        self.d3 = Dropout(0.2)
        self.out = Dense(1, activation="sigmoid")

    def call(self, text):
        x = self.l1(text)
        x = self.d1(x)
        x = self.l2(x)
        x = self.d2(x)
        x = self.l3(x)
        x = self.d3(x)
        x = self.flat(x)
        x = self.out(x)

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

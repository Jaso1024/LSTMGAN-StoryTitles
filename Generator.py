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
import tensorflow_probability as tfp


class ArgmaxLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ArgmaxLayer, self).__init__()

    def call(self, inputs):
      return tf.argmax(inputs, axis=1)

class RecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, sequence_len, vocab_len):
        super(RecurrentLayer, self).__init__()

        self.sequence_len = sequence_len
        self.vocab_len = vocab_len

        self.softmax = Dense(self.vocab_len, activation="softmax")
        self.argmax = ArgmaxLayer()
        self.concat = Concatenate()

    def get_padding(self, iteration):
        num_zeros = self.sequence_len - iteration
        return [0] * num_zeros

    def __call__(self, inputs):
        outs = []
        out_vals = []
            
        for idx in range(self.sequence_len):
            padding = self.get_padding(idx)

            if len(outs) >= 1:
                outs_ = tf.cast(tf.reshape(outs, (1, len(outs))), dtype=tf.float32)
                
                padding = tf.cast(tf.reshape(padding, (1, len(padding))), dtype=tf.float32)

                inp = self.concat([outs_, inputs, padding])
            else:
                padding = tf.cast(tf.reshape(padding, (1, len(padding))), dtype=tf.float32)

                inp = self.concat([inputs, padding])

            output = self.softmax(inp)
            out_vals.append(output)
            dist = tfp.distributions.RelaxedOneHotCategorical(0.1, probs=output)
            sample = dist.sample()
            
            output = tf.compat.v1.distributions.Categorical(probs=sample, dtype=tf.float32)
            outs.append(output.sample())

        return out_vals

class Generator(Model):
    def __init__(self, len_vocab, sequence_len):
        super(Generator, self).__init__()

        self.vocab_len = len_vocab
        self.sequence_len = sequence_len

        self.flat = Flatten()
        

        self.l1 = Dense(256, activation="relu")
        self.l2 = Dense(256, activation="relu")
        
        self.l3 = Dense(256, activation="relu")
        self.l4 = Dense(256, activation="relu")

        self.l5 = RecurrentLayer(self.sequence_len, self.vocab_len)
        
    
    def get_padding(self, iteration):
        num_zeros = self.sequence_len - iteration
        return [0] * num_zeros

    def call(self, noise):

        x = self.l1(noise)
        x = self.l2(x)
        x = self.flat(x)


    
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        
        return x
        

if __name__ == "__main__":

    generator = Generator(5, 5)
    generator.compile(optimizer="adam", loss="mse")
    noise = np.random.randint(low=0, high=1000, size=(1,1,1000))
    print(generator.predict(noise[0]))
    ys = np.random.randint(low=1, high=5, size=(1,5))
    generator.fit(noise,np.array(ys, dtype=np.int32))
    
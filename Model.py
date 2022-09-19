import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten
from tensorflow_text import BertTokenizer
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer

import numpy as np
import os



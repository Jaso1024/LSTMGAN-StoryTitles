from base64 import encode
from hashlib import new
from msilib import sequence
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten, Reshape
from tensorflow_text import BertTokenizer, WordpieceTokenizer
from keras.utils import pad_sequences
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer
import numpy as np
import os
from itertools import repeat
import time
import tensorflow_probability as tfp
from multiprocessing import pool

from Encoder import Encoder
from Generator import Generator
from Discriminator import Discriminator



class GAN(Model):
    def __init__(self, text, maxlen, batch_size = 32) -> None:
        super(GAN, self).__init__()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.encoder = Encoder(text, maxlen)
        self.generator = Generator(self.encoder.vocab_len, self.encoder.sequence_maxlen)
        self.discriminator = Discriminator()

        self.gen_opt = Adam(1e-5)
        self.discrim_opt = Adam(1e-7)

        self.generator.compile(optimizer=self.gen_opt)
        self.discriminator.compile(optimizer=self.discrim_opt)

        self.vocab_len = self.encoder.vocab_len
        self.sequence_len = self.encoder.sequence_maxlen
        self.ckpt, self.ckpt_prefix = self.get_checkpoint()
        self.generator_gradients = []
        self.discriminator_gradients = []
        self.batch_size = batch_size

        print("Vocab length", self.encoder.vocab_len)

    def generate_noise(self):
        return np.random.rand(1,1000)

    @tf.function
    def minmax_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        discriminator_loss = real_loss + fake_loss

        generator_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        return generator_loss, discriminator_loss
    
    def wasserstein_loss(self, real_output, fake_ouptut):
        return -fake_ouptut, (fake_ouptut-real_output)

    def modefied_minmax_loss(self, real_output, fake_output):
        real_loss = tf.math.log(real_output[0][0])
        fake_loss = tf.math.log((1-fake_output[0][0]))
        discriminator_loss = real_loss + fake_loss

        generator_loss = -tf.math.log(fake_output)

        return generator_loss, discriminator_loss
    
    def least_squared_loss(self, real_output, fake_output):
        pass
    
    def get_checkpoint(self):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.gen_opt,
            discriminator_optimizer=self.discrim_opt
        )
        return checkpoint, checkpoint_prefix

    def create_text_representation(self, generator_text, timesteps):
        new_generator_text = []
        marker = 0
        for _ in range(self.sequence_len):
            if marker >= timesteps:
                marker = 0
            new_generator_text.append(generator_text[marker])
            marker += 1
        return new_generator_text
            

    def trainstep(self, real_texts, noises):
        generator_gradients = []
        discriminator_gradients = []
        for real_text, noise in zip(real_texts, noises):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                generator_text = [[0]*self.vocab_len for _ in range(self.sequence_len)]
                for timestep in range(self.sequence_len):
                    fake_text = self.generator(noise, tf.cast(tf.reshape(generator_text, (1, self.sequence_len, self.vocab_len)), tf.float32), training=True)
                    generator_text[timestep] = fake_text[0]

                    #fake_text = tf.cast(fake_text, dtype=tf.float32)
                    #fake_text = tf.reshape(fake_text, (1,self.encoder.sequence_maxlen, self.encoder.vocab_len))
                    #fake_text += 0.05 * tf.random.uniform(tf.shape(fake_text))

                real_text = tf.reshape(real_text, (1,self.encoder.sequence_maxlen, self.encoder.vocab_len))
                real_text = 0.05 * tf.random.uniform(tf.shape(real_text))


                fake_text_y = self.discriminator(tf.reshape(generator_text, (1,self.sequence_len, self.vocab_len)), training=True)
                real_text_y = self.discriminator(real_text, training=True)

                generator_loss, discriminator_loss = self.wasserstein_loss(real_text_y, fake_text_y)
            
                    
                generator_gradients.append(tape1.gradient(generator_loss, self.generator.trainable_variables))
                discriminator_gradients.append(tape2.gradient(discriminator_loss, self.discriminator.trainable_variables))
       

        
        for g_grad, d_grad in zip(generator_gradients, discriminator_gradients):
            
            self.gen_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))
            self.discrim_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
                    
        return generator_loss, discriminator_loss, generator_text, fake_text_y, real_text_y

    def format_tokens(self, tokens):
        new_tokens = np.zeros((self.sequence_len, self.sequence_len, self.vocab_len))
        for timestep in range(new_tokens.shape[0]):
            for idx in range(len(new_tokens)):
                if idx > timestep:
                    break
                new_tokens[timestep][idx] = tokens[0][idx]

        return new_tokens
                
        

    def train(self, text_data, noises, epochs, ckpt_freq=1000, print_freq=1, batch_size=32):
        for epoch in range(1, epochs+1):
            marker = 0

            while True:
                start_time = time.time()
                batch_data_num = 0
                token_batch = []
                noise_batch = []
                if marker >= np.array(noises).shape[0]-1:
                    break

                while True:
                    if batch_data_num >= batch_size:
                        break
                    elif marker >= np.array(noises).shape[0]-1:
                        break

                    noise = np.expand_dims(noises[marker], axis=0)
                    tokens = self.encoder.encode(text_data[marker])

                    if len([token[0] for token in tokens[0]]) > self.encoder.sequence_maxlen:
                        continue
                    



                    token_batch.append(tokens)
                    noise_batch.append(noise)



                    batch_data_num += 1
                    marker += 1



                g_loss, d_loss, generated_text, fdi, rdi = self.trainstep(token_batch, noise_batch)
                generated_text = self.encoder.decode(generated_text)

                if int(marker/batch_size) % print_freq == 0:
                    print(f"""Epoch: {epoch} | Batch: {int(marker/batch_size)} | Time: {time.time()-start_time}\n
                        Generator loss: {g_loss} | Discriminator loss: {d_loss}\n
                        Example text 1: {generated_text}\n
                        {fdi} {rdi}"""
                    )

                if int(marker/batch_size) % ckpt_freq == 0:
                    self.ckpt.save(file_prefix=self.ckpt_prefix)
    
    def generate(self, x):
        sentence = np.zeros((1,self.generator.sequence_len, self.generator.vocab_len))
        for idx in range(self.generator.sequence_len):
            next_word = self.generator(x, sentence)
            sentence[0][idx] = next_word[0]

        return self.encoder.decode(sentence)
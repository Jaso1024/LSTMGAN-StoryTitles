from base64 import encode
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

        self.gen_opt = Adam(1e-4)
        self.discrim_opt = Adam(1e-4)

        self.ckpt, self.ckpt_prefix = self.get_checkpoint()
        self.generator_gradients = []
        self.discriminator_gradients = []
        self.batch_size = batch_size

    def generate_noise(self):
        return np.random.rand(1, 1000)

    @tf.function
    def get_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        discriminator_loss = real_loss + fake_loss

        generator_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        return generator_loss, discriminator_loss
    
    
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
    

    def trainstep(self, real_texts, noises):
        generator_gradients = []
        discriminator_gradients = []

        for real_text, noise in zip(real_texts, noises):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                
                fake_text = self.generator(noise, training=True)

                #new_fake_text = []
                #for token in fake_text:
                #   dist = tfp.distributions.RelaxedOneHotCategorical(.00001, probs=token)
                #    sample = dist.sample()
                #    new_fake_text.append(sample)
                #    print(new_fake_text)

                #fake_text = new_fake_text

                fake_text = tf.cast(fake_text, dtype=tf.float32)
                fake_text = tf.reshape(fake_text, (1,self.encoder.sequence_maxlen, self.encoder.vocab_len))
                

                real_text = [token for token in real_text[0][0]]
                real_text = np.reshape(np.array(real_text), (1,self.encoder.sequence_maxlen, self.encoder.vocab_len))


                fake_text_y = self.discriminator(fake_text, training=True)
                real_text_y = self.discriminator(real_text, training=True)

                generator_loss, discriminator_loss = self.get_loss(real_text_y, fake_text_y)

            generator_gradients.append(tape1.gradient(generator_loss, self.generator.trainable_variables))
            discriminator_gradients.append(tape2.gradient(discriminator_loss, self.discriminator.trainable_variables))
        
        for g_grad, d_grad in zip(generator_gradients, discriminator_gradients):
            self.gen_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))
            self.discrim_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        
        return generator_loss, discriminator_loss

    def train(self, text_data, noises, epochs, ckpt_freq=1000, print_freq=1, batch_size=32):
        for epoch in range(1, epochs+1):
            marker = 0

            while True:
                start_time = time.time()
                batch_data_num = 0
                token_batch = []
                noise_batch = []
                if marker >= np.array(noises).shape[0]:
                    break

                while True:
                    if batch_data_num >= batch_size:
                        break

                    noise = np.expand_dims(noises[marker], axis=0)
                    tokens = self.encoder.encode(text_data[marker])

                    if len([token[0] for token in tokens[0]]) > self.encoder.sequence_maxlen:
                        continue

                    tokens = np.expand_dims(tokens, axis=0)

                    token_batch.append(tokens)
                    noise_batch.append(noise)

                    batch_data_num += 1
                    marker += 1

                g_loss, d_loss = self.trainstep(token_batch, noise_batch)

                if int(marker/batch_size) % print_freq == 0:
                    print(f"Epoch: {epoch} | Batch: {int(marker/batch_size)} | Time: {time.time()-start_time} | Generator loss: {g_loss} | Discriminator loss: {d_loss}")

                if int(marker/batch_size) % ckpt_freq == 0:
                    self.ckpt.save(file_prefix=self.ckpt_prefix)
    
    def generate(self, x1):
        y = self.generator(x1)
        y = tf.cast(y, tf.int64)
        print(y)
        return self.encoder.decode(y)
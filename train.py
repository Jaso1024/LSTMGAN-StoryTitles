from GAN import GAN
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import words
import nltk

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    data = pd.read_csv("nosleep_data.csv")
    categories = data.category.tolist()
    titles = data.title.tolist()
    formatted_titles = []
    words = set(words.words())
    for title in titles:
        try:
            title = title.lower()
        except AttributeError:
            continue
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

    print("Length of dataset: ", len(formatted_titles))
    gan = GAN(formatted_titles, 5, batch_size=32)
    #gan.ae.load_weights("AutoEncoderWeights/aeWeights")
    #gan.save_encoded_data(formatted_titles)
    gan.train_autoencoder(formatted_titles, epochs=100000, batch_size=256)
    gan.ae.save_weights("AutoEncoderWeights/aeWeights")
    

    #groups = np.array(categories)
    #noises = [gan.generate_noise() for _ in range(len(formatted_titles))]
    
    #checkpoint = tf.train.latest_checkpoint("training_checkpoints")
    #print(checkpoint)
    #gan.ckpt.restore(checkpoint)

    #gan.train(formatted_titles, noises, epochs=100, batch_size=128)
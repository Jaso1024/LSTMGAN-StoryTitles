from GAN import GAN
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import words
     

if __name__ == "__main__":
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

    print("Length of dataset: ", len(formatted_titles))
    gan = GAN(formatted_titles, 5, batch_size=32)
    gan.train_autoencoder(formatted_titles, epochs=1000)
    gan.ae.save_weights("AutoEncoderWeights/aeWeights")
    
    noises = [gan.generate_noise() for _ in range(len(formatted_titles))]
    
    gan.train(formatted_titles, noises, epochs=100, batch_size=32)


    
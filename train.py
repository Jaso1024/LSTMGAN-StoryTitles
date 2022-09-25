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
    gan = GAN(formatted_titles, 10, batch_size=2)

    groups = np.array(categories)
    noises = [gan.generate_noise() for _ in range(len(formatted_titles))]
    
    #checkpoint = tf.train.latest_checkpoint("training_checkpoints")
    #print(checkpoint)
    #gan.ckpt.restore(checkpoint)

    

    print("generating")
    test_noise = np.random.randint(low=0, high=1000, size=(1, 1, 1000))
    print(gan.generate(test_noise))
    

    gan.train(formatted_titles, noises, epochs=100, batch_size=32)


    
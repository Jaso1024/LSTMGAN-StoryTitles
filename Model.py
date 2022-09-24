from dis import dis
import re
from tracemalloc import start
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



class Encoder():
    def __init__(self, text, maxlen) -> None:
        self.vocab = self.get_vocab(text)
        self.vocab_len = len(self.vocab)
        lookup_table = StaticVocabularyTable(
            KeyValueTensorInitializer(
                keys=self.vocab,
                key_dtype=tf.string,
                values=tf.range(0, self.vocab_len, dtype=tf.int64)
            ), 
            num_oov_buckets=100
        )
        self.tokenizer = BertTokenizer(lookup_table)

        self.sequence_maxlen = maxlen

    def get_vocab(self, text):
        vocab = set()
        for title in text:
            for word in title.split(" "):
                vocab.add(word)

        return list(vocab) 
    

    def get_maxlen(self, text):
        maxlen = 0
        for title in text:
            current_length = len(self.encode_no_padding(title))
            maxlen = current_length if current_length > maxlen else maxlen
        print(maxlen)
        
        return maxlen

    def pad(self, sequence):
        return pad_sequences(sequence, maxlen=self.sequence_maxlen, padding="post")

    def encode_no_padding(self, text):
        return [token for token in self.tokenizer.tokenize(text)[0]]

    def encode(self, text):
        tokens = [token[0].numpy() for token in self.tokenizer.tokenize(text)[0]]
        tokens = np.array(tokens, dtype=np.float32)
        tokens = self.pad([tokens])
        tokens = tf.expand_dims(tokens, axis=0)
        tokens = np.array(tokens, dtype=np.float32) # Must do this again to prevent tensorflow from removing 0s
        print(tokens)
        return tokens

    def decode(self, tokens):
        tensor = self.tokenizer.detokenize([tokens])   
        return " ".join([word.numpy().decode("utf-8") for word in tensor[0]])

class Generator(Model):
    def __init__(self, len_vocab, sequence_len):
        super(Generator, self).__init__()

        self.vocab_len = len_vocab
        self.sequence_len = sequence_len
        
        self.n1 = Dense(256, activation="relu")
        self.n2 = Dense(256, activation="relu")
        self.n3 = Flatten()
        
        self.g1 = Dense(128, activation="relu", input_shape=(1,1,2))
        self.g2 = Flatten()

        self.concat = Concatenate()
        
        self.fc2 = Dense(256, activation="relu")
        self.fc3 = Dense(256, activation="relu")

        self.out_layer = Dense(self.vocab_len, activation="softmax")
        
    
    def get_padding(self, iteration):
        num_zeros = (self.vocab_len * self.sequence_len) - (self.vocab_len * iteration)
        return [0] * num_zeros

    def call(self, group, noise):

        n = self.n1(noise)
        n = self.n2(n)
        n = self.n3(n)

        g = self.g1(group)
        g = self.g2(g)

        x = self.concat([n, g])
        x = self.fc2(x)
        x = self.fc3(x)
        
        outs = []
        
        for idx in range(self.sequence_len):
            padding = self.get_padding(idx)
            inp = self.concat([*outs, x, *padding])
            outs.append(self.out_layers(inp))
        

        return outs

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.t1 = GRU(256, return_sequences=True)
        self.t2 = GRU(256, return_sequences=True)
        self.t3 = GRU(256)
        self.t3 = Dense(512, activation="relu")
        self.t4 = Flatten()

        self.g1 = Dense(256, activation="relu")
        self.g2 = Flatten()
        
        self.fc1 = Concatenate()
        self.fc2 = Dense(512, activation="relu")
        self.fc3 = Dense(512, activation="relu")
        
        self.out = Dense(1, activation="sigmoid")

    def call(self, text, group):

        t = self.t1(text)
        t = self.t2(t)
        t = self.t3(t)
        t = self.t4(t)
        
        g = self.g1(group)
        g = self.g2(g)

        x = self.fc1([t, g])
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.out(x)

        return x

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
        return np.random.randint(low=0, high=1000, size=(1, 1000))

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


    @tf.function
    def trainstep(self, fake_groups, real_groups, real_texts, noises):
        generator_gradients = []
        discriminator_gradients = []

        for fake_group, real_group, real_text, noise in zip(fake_groups, real_groups, real_texts, noises):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
                distributions = self.generator(fake_group, noise, training=True)
                fake_text = []
                for dist in distributions:
                    fake_text.append(np.argmax(dist))

                fake_text = tf.expand_dims(fake_text, axis=0)

                real_text_y = self.discriminator(real_text, real_group, training=True)
                fake_text_y = self.discriminator(fake_text, fake_group, training=True)

                generator_loss, discriminator_loss = self.get_loss(real_text_y, fake_text_y)

            generator_gradients.append(tape1.gradient(generator_loss, self.generator.trainable_variables))
            discriminator_gradients.append(tape2.gradient(discriminator_loss, self.discriminator.trainable_variables))
        
        for g_grad, d_grad in zip(generator_gradients, discriminator_gradients):
            self.gen_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))
            self.discrim_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        
        return generator_loss, discriminator_loss


    def train(self, fake_groups, real_groups, text_data, noises, epochs, ckpt_freq=10, batch_size=32):
        for epoch in range(1, epochs+1):
            marker = 0
            fake_group_batch, real_group_batch, token_batch, noise_batch = [[],[],[],[]]
            start_time = time.time()
            for _ in range(batch_size):
                if marker >= len(noises):
                    break

                fake_group = np.expand_dims(fake_groups[marker], axis=0)
                fake_group = np.expand_dims(fake_groups[marker], axis=0)
                real_group = np.expand_dims(real_groups[marker], axis=0)
                noise = np.expand_dims(noises[marker], axis=0)
                tokens = self.encoder.encode(text_data[marker])
                tokens = np.expand_dims(tokens, axis=0)

                fake_group_batch.append(fake_group)
                real_group_batch.append(real_group)
                token_batch.append(tokens)
                noise_batch.append(noise)

                marker += 1


            g_loss, d_loss = self.trainstep(fake_group_batch, real_group_batch, token_batch, noise_batch)

            print(f"Epoch: {epoch} | Time: {time.time()-start_time} | Generator loss: {g_loss} | Discriminator loss: {d_loss}")

            if epoch % ckpt_freq == 0:
                self.ckpt.save(file_prefix=self.ckpt_prefix)
    
    def generate(self, x1, x2):
        y = self.generator(x1, x2)
        y = tf.cast(y, tf.int64)
        print(y)
        return self.encoder.decode(y)
    
def test_encoder():
    text = "Hello my name is BERT I am an AI used for NLP".lower()
    encoder = Encoder([text])
    tokens = encoder.tokenizer.tokenize("i am bert")
    print(tokens)
    print(encoder.tokenizer.detokenize(tokens))

def test_generator():
    text = "A period of unrest and civil wars in the 1st century BCE marked the transition of Rome from a republic to an empire. This period encompassed the career of Julius Caesar, who eventually took full power over Rome as its dictator. After his assassination in 44 BCE, the triumvirate of Mark Antony, Lepidus, and Octavian, Caesar’s nephew, ruled. It was not long before Octavian went to war against Antony in northern Africa, and after his victory at Actium (31 BCE) he was crowned Rome’s first emperor, Augustus. His reign, from 27 BCE to 14 CE, was distinguished by stability and peace. Augustus established a form of government known as a principate, which combined some elements from the republic with the traditional powers of a monarchy. The Senate still functioned, though Augustus, as princeps, or first citizen, remained in control of the government. With a mind toward maintaining the structure of power entrusted to his rule, Augustus began thinking early about who should follow him. Death played havoc with his attempts to select his successor. He had no son and his nephew Marcellus, his son-in-law Agrippa, and his grandsons Gaius and Lucius each predeceased him. He eventually chose Tiberius, a scion of the ultra-aristocratic Claudia gens, and in 4 CE adopted him as his son. Tiberius (reigned 14–37) became the first successor in the Julio-Claudian dynasty and ruled as an able administrator but cruel tyrant. His great-nephew Caligula (37–41) reigned as an absolutist, his short reign filled with reckless spending, callous murders, and humiliation of the Senate. Claudius (41–54) centralized state finances in the imperial household, thus making rapid strides in organizing the imperial bureaucracy, but was ruthless toward the senators and equites. Nero (54–68) left administration to capable advisers for a few years but then asserted himself as a vicious despot. He brought the dynasty to its end by being the first emperor to suffer damnatio memoriae: his reign was officially stricken from the record by order of the Senate. Following a war of succession, Vespasian became emperor, and the Flavian dynasty was established. His reign (69–79) was noted for his reorganization of the army, making it more loyal and professional; for his expansion of the membership of the Senate, bringing in administrators with a sense of service; for his increase and systematization of taxation; and for his strengthening of the frontiers of the empire (though little new territory was added). The brief but popular reign of his son Titus (79–81) was followed by the autocracy of Domitian (81–96), Vespasian’s other son, who fought the senatorial class and instituted taxes and confiscations for costly buildings, games, and shows. A reign of terror in his final years was ended by his assassination. The Flavian dynasty, like the Julio-Claudian, ended with an emperor whose memory was officially damned. Domitian was succeeded by an elderly senator of some distinction, Marcus Cocceius Nerva (96–98). Among the beloved rulers of Rome that succeeded him were Trajan (reigned 98–117), Hadrian (117–138), Antoninus Pius (138–161), and Marcus Aurelius (161–180). Together these are known as the Five Good Emperors. Their non-hereditary succession oversaw a golden age, which witnessed a considerable amount of expansion and consolidation. But all the changes that occurred during this era, beneficial as they were, brought with them the attendant evils of excessive centralization. The concentration of an empire in the hands of an emperor like Commodus (180–192)—juvenile, incompetent, and decadent—was enough to steer it toward decline. The following century was plagued by strife and mismanagement. When the commander of the Danube army, Septimius Severus, was swept to power in 193, he effectively made Rome a military monarchy. The “barbarian invasions” weighed heavily on the empire, as did usurpations and political destabilization. The instability fed on itself and was responsible for heavy expenditure of both life and treasure. Disruptions in commerce, harsh taxation, inflation, and extortion from stationed troops all contributed to perpetual economic hardship for decades. A period of recovery began with Diocletian (284–305), whose broad reforms renewed the integrity and cohesion of the imperial administration. His most notable adjustment was the reorganization of the empire into a tetrarchy, wherein power was divided among himself, Maximian (who became Augustus, or emperor, in 286), Constantius (who became Caesar, or hereditary prince, in 293), and Galerius (who also became Caesar in 293). The arrangement proved practical in stabilizing the empire for a time against usurpation, and it also promised the rulers legitimacy and regular succession. The tetrarchy soon led to confusion, however, and by 308 there were seven pretenders to the title of Augustus. Among them was Constantius’s eldest son, Constantine, who was passed over for formal succession. As a high-ranking military tribune, however, he had a forceful command and was able to eliminate his rivals successively in the West. He became the uncontested emperor of the West in 312 and, upon the defeat of his co-emperor in the East, he became the sole Augustus of the empire in 324."
    text = text.split(". ")
    encoder = Encoder(text)

    gen = Generator(encoder.vocab_len, encoder.sequence_maxlen)
    noise = np.random.randint(low=0, high=1000, size=(1, 1000))
    output = gen(np.array([[0,1]]), noise)
    tokens = []
    for dist in output:
        tokens.append(np.argmax(dist))
        
    print(tokens)
    print(encoder.decode(tokens))
    

def test_discriminator():
    text = "A period of unrest and civil wars in the 1st century BCE marked the transition of Rome from a republic to an empire. This period encompassed the career of Julius Caesar, who eventually took full power over Rome as its dictator. After his assassination in 44 BCE, the triumvirate of Mark Antony, Lepidus, and Octavian, Caesar’s nephew, ruled. It was not long before Octavian went to war against Antony in northern Africa, and after his victory at Actium (31 BCE) he was crowned Rome’s first emperor, Augustus. His reign, from 27 BCE to 14 CE, was distinguished by stability and peace. Augustus established a form of government known as a principate, which combined some elements from the republic with the traditional powers of a monarchy. The Senate still functioned, though Augustus, as princeps, or first citizen, remained in control of the government. With a mind toward maintaining the structure of power entrusted to his rule, Augustus began thinking early about who should follow him. Death played havoc with his attempts to select his successor. He had no son and his nephew Marcellus, his son-in-law Agrippa, and his grandsons Gaius and Lucius each predeceased him. He eventually chose Tiberius, a scion of the ultra-aristocratic Claudia gens, and in 4 CE adopted him as his son. Tiberius (reigned 14–37) became the first successor in the Julio-Claudian dynasty and ruled as an able administrator but cruel tyrant. His great-nephew Caligula (37–41) reigned as an absolutist, his short reign filled with reckless spending, callous murders, and humiliation of the Senate. Claudius (41–54) centralized state finances in the imperial household, thus making rapid strides in organizing the imperial bureaucracy, but was ruthless toward the senators and equites. Nero (54–68) left administration to capable advisers for a few years but then asserted himself as a vicious despot. He brought the dynasty to its end by being the first emperor to suffer damnatio memoriae: his reign was officially stricken from the record by order of the Senate. Following a war of succession, Vespasian became emperor, and the Flavian dynasty was established. His reign (69–79) was noted for his reorganization of the army, making it more loyal and professional; for his expansion of the membership of the Senate, bringing in administrators with a sense of service; for his increase and systematization of taxation; and for his strengthening of the frontiers of the empire (though little new territory was added). The brief but popular reign of his son Titus (79–81) was followed by the autocracy of Domitian (81–96), Vespasian’s other son, who fought the senatorial class and instituted taxes and confiscations for costly buildings, games, and shows. A reign of terror in his final years was ended by his assassination. The Flavian dynasty, like the Julio-Claudian, ended with an emperor whose memory was officially damned. Domitian was succeeded by an elderly senator of some distinction, Marcus Cocceius Nerva (96–98). Among the beloved rulers of Rome that succeeded him were Trajan (reigned 98–117), Hadrian (117–138), Antoninus Pius (138–161), and Marcus Aurelius (161–180). Together these are known as the Five Good Emperors. Their non-hereditary succession oversaw a golden age, which witnessed a considerable amount of expansion and consolidation. But all the changes that occurred during this era, beneficial as they were, brought with them the attendant evils of excessive centralization. The concentration of an empire in the hands of an emperor like Commodus (180–192)—juvenile, incompetent, and decadent—was enough to steer it toward decline. The following century was plagued by strife and mismanagement. When the commander of the Danube army, Septimius Severus, was swept to power in 193, he effectively made Rome a military monarchy. The “barbarian invasions” weighed heavily on the empire, as did usurpations and political destabilization. The instability fed on itself and was responsible for heavy expenditure of both life and treasure. Disruptions in commerce, harsh taxation, inflation, and extortion from stationed troops all contributed to perpetual economic hardship for decades. A period of recovery began with Diocletian (284–305), whose broad reforms renewed the integrity and cohesion of the imperial administration. His most notable adjustment was the reorganization of the empire into a tetrarchy, wherein power was divided among himself, Maximian (who became Augustus, or emperor, in 286), Constantius (who became Caesar, or hereditary prince, in 293), and Galerius (who also became Caesar in 293). The arrangement proved practical in stabilizing the empire for a time against usurpation, and it also promised the rulers legitimacy and regular succession. The tetrarchy soon led to confusion, however, and by 308 there were seven pretenders to the title of Augustus. Among them was Constantius’s eldest son, Constantine, who was passed over for formal succession. As a high-ranking military tribune, however, he had a forceful command and was able to eliminate his rivals successively in the West. He became the uncontested emperor of the West in 312 and, upon the defeat of his co-emperor in the East, he became the sole Augustus of the empire in 324."
    text = text.split(". ")
    encoder = Encoder(text)

    gen = Generator(encoder.vocab_len, encoder.sequence_maxlen)
    noise = np.random.randint(low=0, high=1000, size=(1, 1000))
    output = gen(np.array([[0,1]]), noise)
    tokens = []
    for dist in output:
        tokens.append(np.argmax(dist))
    tokens = encoder.encode(encoder.decode(tokens))
    dis = Discriminator()
    print(dis([tokens], np.array([[0,1]])))

def test_gan():
    text = "A period of unrest and civil wars in the 1st century BCE marked the transition of Rome from a republic to an empire. This period encompassed the career of Julius Caesar, who eventually took full power over Rome as its dictator. After his assassination in 44 BCE, the triumvirate of Mark Antony, Lepidus, and Octavian, Caesar’s nephew, ruled. It was not long before Octavian went to war against Antony in northern Africa, and after his victory at Actium (31 BCE) he was crowned Rome’s first emperor, Augustus. His reign, from 27 BCE to 14 CE, was distinguished by stability and peace. Augustus established a form of government known as a principate, which combined some elements from the republic with the traditional powers of a monarchy. The Senate still functioned, though Augustus, as princeps, or first citizen, remained in control of the government. With a mind toward maintaining the structure of power entrusted to his rule, Augustus began thinking early about who should follow him. Death played havoc with his attempts to select his successor. He had no son and his nephew Marcellus, his son-in-law Agrippa, and his grandsons Gaius and Lucius each predeceased him. He eventually chose Tiberius, a scion of the ultra-aristocratic Claudia gens, and in 4 CE adopted him as his son. Tiberius (reigned 14–37) became the first successor in the Julio-Claudian dynasty and ruled as an able administrator but cruel tyrant. His great-nephew Caligula (37–41) reigned as an absolutist, his short reign filled with reckless spending, callous murders, and humiliation of the Senate. Claudius (41–54) centralized state finances in the imperial household, thus making rapid strides in organizing the imperial bureaucracy, but was ruthless toward the senators and equites. Nero (54–68) left administration to capable advisers for a few years but then asserted himself as a vicious despot. He brought the dynasty to its end by being the first emperor to suffer damnatio memoriae: his reign was officially stricken from the record by order of the Senate. Following a war of succession, Vespasian became emperor, and the Flavian dynasty was established. His reign (69–79) was noted for his reorganization of the army, making it more loyal and professional; for his expansion of the membership of the Senate, bringing in administrators with a sense of service; for his increase and systematization of taxation; and for his strengthening of the frontiers of the empire (though little new territory was added). The brief but popular reign of his son Titus (79–81) was followed by the autocracy of Domitian (81–96), Vespasian’s other son, who fought the senatorial class and instituted taxes and confiscations for costly buildings, games, and shows. A reign of terror in his final years was ended by his assassination. The Flavian dynasty, like the Julio-Claudian, ended with an emperor whose memory was officially damned. Domitian was succeeded by an elderly senator of some distinction, Marcus Cocceius Nerva (96–98). Among the beloved rulers of Rome that succeeded him were Trajan (reigned 98–117), Hadrian (117–138), Antoninus Pius (138–161), and Marcus Aurelius (161–180). Together these are known as the Five Good Emperors. Their non-hereditary succession oversaw a golden age, which witnessed a considerable amount of expansion and consolidation. But all the changes that occurred during this era, beneficial as they were, brought with them the attendant evils of excessive centralization. The concentration of an empire in the hands of an emperor like Commodus (180–192)—juvenile, incompetent, and decadent—was enough to steer it toward decline. The following century was plagued by strife and mismanagement. When the commander of the Danube army, Septimius Severus, was swept to power in 193, he effectively made Rome a military monarchy. The “barbarian invasions” weighed heavily on the empire, as did usurpations and political destabilization. The instability fed on itself and was responsible for heavy expenditure of both life and treasure. Disruptions in commerce, harsh taxation, inflation, and extortion from stationed troops all contributed to perpetual economic hardship for decades. A period of recovery began with Diocletian (284–305), whose broad reforms renewed the integrity and cohesion of the imperial administration. His most notable adjustment was the reorganization of the empire into a tetrarchy, wherein power was divided among himself, Maximian (who became Augustus, or emperor, in 286), Constantius (who became Caesar, or hereditary prince, in 293), and Galerius (who also became Caesar in 293). The arrangement proved practical in stabilizing the empire for a time against usurpation, and it also promised the rulers legitimacy and regular succession. The tetrarchy soon led to confusion, however, and by 308 there were seven pretenders to the title of Augustus. Among them was Constantius’s eldest son, Constantine, who was passed over for formal succession. As a high-ranking military tribune, however, he had a forceful command and was able to eliminate his rivals successively in the West. He became the uncontested emperor of the West in 312 and, upon the defeat of his co-emperor in the East, he became the sole Augustus of the empire in 324.".lower()
    groups = np.array([[0,1], [0,1], [1,0]], dtype=np.float32)
    real_text = ["civil wars", "transition of rome from a republic to an empire", "julius caesar"]
    gan = GAN(text, 10)
    noises = np.array([gan.generate_noise(), gan.generate_noise(), gan.generate_noise()])
    gan.train(groups, groups, real_text, noises, epochs=1000)
    test_group = np.array([[1,0]])
    test_noise = np.random.randint(low=0, high=1000, size=(1, 1, 1000))
    thing = gan.generator([test_group, test_noise])
    thing = thing
    thing = tf.cast(thing, tf.int64)
    print(thing) 
    print(gan.encoder.decode(thing))

if __name__ == "__main__":
    #test_encoder()
    #test_generator()
    #test_discriminator()
    pass

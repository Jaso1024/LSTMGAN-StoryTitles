from base64 import encode
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Concatenate, LSTM, Embedding, GRU, InputLayer, Flatten
from tensorflow_text import BertTokenizer
from keras.utils import pad_sequences
from tensorflow.lookup import StaticVocabularyTable, KeyValueTensorInitializer

import numpy as np
import os


class Encoder():
    def __init__(self, text, maxlen) -> None:
        self.vocab = self.get_vocab(text)
        self.sequence_maxlen = maxlen
        lookup_table = StaticVocabularyTable(
            KeyValueTensorInitializer(
                keys=self.vocab,
                key_dtype=tf.string,
                values=tf.range(1, len(self.vocab)+1, dtype=tf.int64)
            ), 
            num_oov_buckets=100
        )
        self.tokenizer = BertTokenizer(lookup_table)


    def pad(self, sequence):
        padded_sequence = pad_sequences(sequence, maxlen=self.sequence_maxlen, padding='post')
        return padded_sequence

    def get_vocab(self, text):
        text = text.split(" ")
        return list(set(text)) 
    
    def get_vocab_length(self):
        return len(self.vocab)
    
    def encode(self, text):
        text_length = len(text.split(" "))
        tokens = self.tokenizer.tokenize(text)
        tokens = tf.reshape(tokens, (1, text_length))
        tokens = self.pad(tokens)
        return tokens

    def decode(self, tokens):
        return self.tokenizer.detokenize(tokens)    

class Generator(Model):
    def __init__(self, maxlen):
        super(Generator, self).__init__()

        self.n1 = Dense(512, activation="relu")
        self.n2 = Dense(256, activation="relu")
        self.n3 = Flatten()
        
        self.g1 = Dense(256, activation="relu")
        self.g2 = Flatten()

        self.fc1 = Concatenate()
        self.fc2 = Dense(256, activation="relu")
        self.fc3 = Dense(256, activation="relu")

        self.out = Dense(maxlen, activation="relu")
    
    def generate_noise(self):
        return np.random.randint(low=0, high=1000, size=(1, 1000))

    def call(self, inp):
        noise = self.generate_noise()
        n = self.n1(noise)
        n = self.n2(n)
        n = self.n3(n)

        g = self.g1(inp)
        g = self.g2(g)

        x = self.fc1([n, g])
        x = self.fc2(x)
        x = self.fc3(x)
        
        x = self.out(x)

        return x

class Discriminator(Model):
    def __init__(self, num_uniques):
        super(Discriminator, self).__init__()

        self.t1 = Embedding(num_uniques, 128)
        self.t2 = GRU(512, return_sequences=True)
        self.t3 = GRU(512, return_sequences=True)
        self.t4 = Dense(256, activation="relu")
        self.t5 = Flatten()

        self.g1 = Dense(256, activation="relu")
        self.g2 = Flatten()
        
        self.fc1 = Concatenate()
        self.fc2 = Dense(512, activation="relu")
        self.fc3 = Dense(128, activation="relu")
        
        self.out = Dense(1)

    def call(self, inp):
        text = inp[0]
        group = inp[1]

        t = self.t1(text)
        t = self.t2(t)
        t = self.t3(t)
        t = self.t4(t)
        t = self.t5(t)
        
        g = self.g1(group)
        g = self.g2(g)

        x = self.fc1([t, g])
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.out(x)

        return x

class GAN():
    def __init__(self, text, maxlen) -> None:
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.encoder = Encoder(text, maxlen)
        vocab = self.encoder.get_vocab_length()
        self.generator = Generator(maxlen)
        self.discriminator = Discriminator(vocab)

        self.gen_opt = Adam(1e-4)
        self.discrim_opt = Adam(1e-4)

        self.ckpt, self.ckpt_prefix = self.get_checkpoint()

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
    def trainstep(self, fake_group, real_group, real_text):

        with tf.GradientTape(persistent=True) as tape1, tf.GradientTape() as tape2:
            fake_text = self.generator(fake_group, training=True)

            real_text_y = self.discriminator([real_text, real_group], training=True)
            fake_text_y = self.discriminator([fake_text, fake_group], training=True)

            generator_loss, discriminator_loss = self.get_loss(real_text_y, fake_text_y)

        generator_grad = tape1.gradient(generator_loss, self.generator.trainable_variables, unconnected_gradients='zero')
        discriminator_grad = tape2.gradient(discriminator_loss, self.discriminator.trainable_variables)
        print(generator_grad)
        self.gen_opt.apply_gradients(zip(generator_grad, self.generator.trainable_variables))
        self.discrim_opt.apply_gradients(zip(discriminator_grad, self.discriminator.trainable_variables))
    
    def train(self, fake_groups, real_groups, text_data, epochs, ckpt_freq=15):
        for epoch in range(1, epochs+1):
            for fake_group, real_group, text in zip(fake_groups, real_groups, text_data):
                fake_group = np.expand_dims(fake_group, axis=0)
                real_group = np.expand_dims(real_group, axis=0)
                tokens = self.encoder.encode(text)
                self.trainstep(fake_group, real_group, tokens)

            if epoch % ckpt_freq == 0:
                self.ckpt.save(file_prefix=self.ckpt_prefix)
    

def test_encoder():
    text = "Hello my name is BERT I am an AI used for NLP"
    encoder = Encoder(text)
    tokens = encoder.tokenizer.tokenize("I am BERT")
    print(tokens)
    print(encoder.tokenizer.detokenize(tokens))

def test_generator():
    gen = Generator()
    thing = gen.predict(np.array([[0,1]]))
    text = "A period of unrest and civil wars in the 1st century BCE marked the transition of Rome from a republic to an empire. This period encompassed the career of Julius Caesar, who eventually took full power over Rome as its dictator. After his assassination in 44 BCE, the triumvirate of Mark Antony, Lepidus, and Octavian, Caesar’s nephew, ruled. It was not long before Octavian went to war against Antony in northern Africa, and after his victory at Actium (31 BCE) he was crowned Rome’s first emperor, Augustus. His reign, from 27 BCE to 14 CE, was distinguished by stability and peace. Augustus established a form of government known as a principate, which combined some elements from the republic with the traditional powers of a monarchy. The Senate still functioned, though Augustus, as princeps, or first citizen, remained in control of the government. With a mind toward maintaining the structure of power entrusted to his rule, Augustus began thinking early about who should follow him. Death played havoc with his attempts to select his successor. He had no son and his nephew Marcellus, his son-in-law Agrippa, and his grandsons Gaius and Lucius each predeceased him. He eventually chose Tiberius, a scion of the ultra-aristocratic Claudia gens, and in 4 CE adopted him as his son. Tiberius (reigned 14–37) became the first successor in the Julio-Claudian dynasty and ruled as an able administrator but cruel tyrant. His great-nephew Caligula (37–41) reigned as an absolutist, his short reign filled with reckless spending, callous murders, and humiliation of the Senate. Claudius (41–54) centralized state finances in the imperial household, thus making rapid strides in organizing the imperial bureaucracy, but was ruthless toward the senators and equites. Nero (54–68) left administration to capable advisers for a few years but then asserted himself as a vicious despot. He brought the dynasty to its end by being the first emperor to suffer damnatio memoriae: his reign was officially stricken from the record by order of the Senate. Following a war of succession, Vespasian became emperor, and the Flavian dynasty was established. His reign (69–79) was noted for his reorganization of the army, making it more loyal and professional; for his expansion of the membership of the Senate, bringing in administrators with a sense of service; for his increase and systematization of taxation; and for his strengthening of the frontiers of the empire (though little new territory was added). The brief but popular reign of his son Titus (79–81) was followed by the autocracy of Domitian (81–96), Vespasian’s other son, who fought the senatorial class and instituted taxes and confiscations for costly buildings, games, and shows. A reign of terror in his final years was ended by his assassination. The Flavian dynasty, like the Julio-Claudian, ended with an emperor whose memory was officially damned. Domitian was succeeded by an elderly senator of some distinction, Marcus Cocceius Nerva (96–98). Among the beloved rulers of Rome that succeeded him were Trajan (reigned 98–117), Hadrian (117–138), Antoninus Pius (138–161), and Marcus Aurelius (161–180). Together these are known as the Five Good Emperors. Their non-hereditary succession oversaw a golden age, which witnessed a considerable amount of expansion and consolidation. But all the changes that occurred during this era, beneficial as they were, brought with them the attendant evils of excessive centralization. The concentration of an empire in the hands of an emperor like Commodus (180–192)—juvenile, incompetent, and decadent—was enough to steer it toward decline. The following century was plagued by strife and mismanagement. When the commander of the Danube army, Septimius Severus, was swept to power in 193, he effectively made Rome a military monarchy. The “barbarian invasions” weighed heavily on the empire, as did usurpations and political destabilization. The instability fed on itself and was responsible for heavy expenditure of both life and treasure. Disruptions in commerce, harsh taxation, inflation, and extortion from stationed troops all contributed to perpetual economic hardship for decades. A period of recovery began with Diocletian (284–305), whose broad reforms renewed the integrity and cohesion of the imperial administration. His most notable adjustment was the reorganization of the empire into a tetrarchy, wherein power was divided among himself, Maximian (who became Augustus, or emperor, in 286), Constantius (who became Caesar, or hereditary prince, in 293), and Galerius (who also became Caesar in 293). The arrangement proved practical in stabilizing the empire for a time against usurpation, and it also promised the rulers legitimacy and regular succession. The tetrarchy soon led to confusion, however, and by 308 there were seven pretenders to the title of Augustus. Among them was Constantius’s eldest son, Constantine, who was passed over for formal succession. As a high-ranking military tribune, however, he had a forceful command and was able to eliminate his rivals successively in the West. He became the uncontested emperor of the West in 312 and, upon the defeat of his co-emperor in the East, he became the sole Augustus of the empire in 324."
    encoder = Encoder(text)
    thing = tf.cast(thing, tf.int64)
    print(thing) 
    print(encoder.decode(thing))

def test_discriminator():
    gen = Generator()
    text = gen.predict(np.array([[0,1]]))
    dis = Discriminator(500)
    print(dis.predict([text, np.array([[0,1]])]))

if __name__ == "__main__":
    text = "A period of unrest and civil wars in the 1st century BCE marked the transition of Rome from a republic to an empire. This period encompassed the career of Julius Caesar, who eventually took full power over Rome as its dictator. After his assassination in 44 BCE, the triumvirate of Mark Antony, Lepidus, and Octavian, Caesar’s nephew, ruled. It was not long before Octavian went to war against Antony in northern Africa, and after his victory at Actium (31 BCE) he was crowned Rome’s first emperor, Augustus. His reign, from 27 BCE to 14 CE, was distinguished by stability and peace. Augustus established a form of government known as a principate, which combined some elements from the republic with the traditional powers of a monarchy. The Senate still functioned, though Augustus, as princeps, or first citizen, remained in control of the government. With a mind toward maintaining the structure of power entrusted to his rule, Augustus began thinking early about who should follow him. Death played havoc with his attempts to select his successor. He had no son and his nephew Marcellus, his son-in-law Agrippa, and his grandsons Gaius and Lucius each predeceased him. He eventually chose Tiberius, a scion of the ultra-aristocratic Claudia gens, and in 4 CE adopted him as his son. Tiberius (reigned 14–37) became the first successor in the Julio-Claudian dynasty and ruled as an able administrator but cruel tyrant. His great-nephew Caligula (37–41) reigned as an absolutist, his short reign filled with reckless spending, callous murders, and humiliation of the Senate. Claudius (41–54) centralized state finances in the imperial household, thus making rapid strides in organizing the imperial bureaucracy, but was ruthless toward the senators and equites. Nero (54–68) left administration to capable advisers for a few years but then asserted himself as a vicious despot. He brought the dynasty to its end by being the first emperor to suffer damnatio memoriae: his reign was officially stricken from the record by order of the Senate. Following a war of succession, Vespasian became emperor, and the Flavian dynasty was established. His reign (69–79) was noted for his reorganization of the army, making it more loyal and professional; for his expansion of the membership of the Senate, bringing in administrators with a sense of service; for his increase and systematization of taxation; and for his strengthening of the frontiers of the empire (though little new territory was added). The brief but popular reign of his son Titus (79–81) was followed by the autocracy of Domitian (81–96), Vespasian’s other son, who fought the senatorial class and instituted taxes and confiscations for costly buildings, games, and shows. A reign of terror in his final years was ended by his assassination. The Flavian dynasty, like the Julio-Claudian, ended with an emperor whose memory was officially damned. Domitian was succeeded by an elderly senator of some distinction, Marcus Cocceius Nerva (96–98). Among the beloved rulers of Rome that succeeded him were Trajan (reigned 98–117), Hadrian (117–138), Antoninus Pius (138–161), and Marcus Aurelius (161–180). Together these are known as the Five Good Emperors. Their non-hereditary succession oversaw a golden age, which witnessed a considerable amount of expansion and consolidation. But all the changes that occurred during this era, beneficial as they were, brought with them the attendant evils of excessive centralization. The concentration of an empire in the hands of an emperor like Commodus (180–192)—juvenile, incompetent, and decadent—was enough to steer it toward decline. The following century was plagued by strife and mismanagement. When the commander of the Danube army, Septimius Severus, was swept to power in 193, he effectively made Rome a military monarchy. The “barbarian invasions” weighed heavily on the empire, as did usurpations and political destabilization. The instability fed on itself and was responsible for heavy expenditure of both life and treasure. Disruptions in commerce, harsh taxation, inflation, and extortion from stationed troops all contributed to perpetual economic hardship for decades. A period of recovery began with Diocletian (284–305), whose broad reforms renewed the integrity and cohesion of the imperial administration. His most notable adjustment was the reorganization of the empire into a tetrarchy, wherein power was divided among himself, Maximian (who became Augustus, or emperor, in 286), Constantius (who became Caesar, or hereditary prince, in 293), and Galerius (who also became Caesar in 293). The arrangement proved practical in stabilizing the empire for a time against usurpation, and it also promised the rulers legitimacy and regular succession. The tetrarchy soon led to confusion, however, and by 308 there were seven pretenders to the title of Augustus. Among them was Constantius’s eldest son, Constantine, who was passed over for formal succession. As a high-ranking military tribune, however, he had a forceful command and was able to eliminate his rivals successively in the West. He became the uncontested emperor of the West in 312 and, upon the defeat of his co-emperor in the East, he became the sole Augustus of the empire in 324.".lower()
    groups = np.array([[0,1], [0,1], [1,0]])
    real_text = ["civil wars", "transition of rome from a republic to an empire", "julius caesar"]
    gan = GAN(text, 100)
    gan.train(groups, groups, real_text, epochs=5)
    thing = gan.generator.predict(np.array([[0,1]]))
    thing = tf.cast(thing, tf.int64)
    print(thing) 
    print(gan.encoder.decode(thing))

    

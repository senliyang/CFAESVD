import numpy as np
import tensorflow as tf
from keras import layers, models
from numpy import matlib as nm
import pandas as pd
from sklearn import preprocessing
from keras import utils


def data_process(d_sim,new_association):


    R_A = np.repeat(new_association, repeats=134, axis=0)#(157718,134)
    # print(R_A.shape)
    sd = nm.repmat(d_sim, 1177, 1)#(157718,134)
    # print(sd.shape)

    train1 = np.concatenate((R_A, sd), axis=1)
    label = new_association.reshape((157718, 1))

    return train1, label

def disease_auto_encoder(y_train):
    seed = 2024
    np.random.seed(seed)
    tf.random.set_seed(seed)
    encoding_dim = 64
    input_vector = layers.Input(shape=(268,))

    # encoder layer
    encoded = layers.Dense(250, activation='relu')(input_vector)
    encoded = layers.Dense(150, activation='relu')(encoded)
    encoded = layers.Dense(100, activation='relu')(encoded)
    disease_encoder_output = layers.Dense(encoding_dim)(encoded)

    # decoder layer
    decoded = layers.Dense(100, activation='relu')(disease_encoder_output)
    decoded = layers.Dense(150, activation='relu')(decoded)
    decoded = layers.Dense(250, activation='relu')(decoded)
    decoded = layers.Dense(268, activation='tanh')(decoded)


    # build a autoencoder model
    autoencoder = models.Model(inputs=input_vector, outputs=decoded)
    encoder = models.Model(inputs=input_vector, outputs=disease_encoder_output)

    # activate model
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(y_train, y_train, epochs=20, batch_size=100, shuffle=True)
    disease_encoded_vector = encoder.predict(y_train)
    return disease_encoded_vector


def four_AE(d_sim, new_association):

    dtrain, label = data_process(d_sim, new_association)
    d_features = disease_auto_encoder(dtrain)
    return d_features







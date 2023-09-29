#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the images to the range of [0., 1.]
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Reshape the data to 4D tensors
X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1))


for latent_dim in range(28, 30, 2):
    print("#############################################")
    print("Latent dimension: ", latent_dim)
    print("#############################################")
    encoder = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, strides=2, activation="relu", padding="same"),
            layers.Conv2D(64, 3, strides=2, activation="relu", padding="same"),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(latent_dim, name="latent_vector"),
        ],
        name="encoder",
    )

    decoder = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 32, activation="relu"),
            layers.Reshape((7, 7, 32)),
            layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
        ],
        name="decoder",
    )

    ae = keras.Sequential([encoder, decoder], name="ae")

    ae.compile(
        loss="binary_crossentropy", 
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    history = ae.fit(
        X_train, X_train, 
        epochs=5, 
        validation_data=(X_test, X_test)
    )

    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    fig.savefig("accuracies/accuracies for latent_dim = " + str(latent_dim) + ".png")


    # Plot the latent space
    latent_vectors = encoder.predict(X_test)
    fig, ax = plt.subplots()
    plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=y_test)
    plt.colorbar()
    plt.set_cmap("viridis")
    fig.savefig("latent_space/latent space for latent_dim = " + str(latent_dim) + ".png")
    
    # Plot the reconstructed images
    n = 10
    fig, ax = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        # Display original
        ax[0][i].imshow(X_test[i].reshape(28, 28))
        ax[0][i].get_xaxis().set_visible(False)
        ax[0][i].get_yaxis().set_visible(False)

        # Display reconstruction
        ax[1][i].imshow(ae.predict(X_test)[i].reshape(28, 28))
        ax[1][i].get_xaxis().set_visible(False)
        ax[1][i].get_yaxis().set_visible(False)
    plt.set_cmap("gray")
    fig.savefig("numbers/reconstructed images for latent_dim = " + str(latent_dim) + ".png")

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
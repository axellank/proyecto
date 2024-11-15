from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

def create_model(hidden_neurons=128, activation="relu"):
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Flatten(),
        Dense(hidden_neurons, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

import numpy as np
from models.base_model import create_model
from data.load_data import load_fashion_mnist
import csv

(x_train, y_train), (x_test, y_test) = load_fashion_mnist()

def random_search(iterations=50):
    history = []

    for i in range(iterations):
        hidden_neurons = np.random.randint(50, 200)
        activation = np.random.choice(["relu", "tanh"])
        model = create_model(hidden_neurons=hidden_neurons, activation=activation)
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        history.append((hidden_neurons, activation, accuracy))

    with open("results/random_search.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Hidden Neurons", "Activation", "Accuracy"])
        writer.writerows(history)

    return max(history, key=lambda x: x[2])

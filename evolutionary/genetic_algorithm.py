import numpy as np
from models.base_model import create_model
from data.load_data import load_fashion_mnist
import csv

(x_train, y_train), (x_test, y_test) = load_fashion_mnist()

def evaluate_model(hidden_neurons, activation):
    model = create_model(hidden_neurons=hidden_neurons, activation=activation)
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

def genetic_algorithm(pop_size=10, generations=5, mutation_rate=0.1):
    population = [{"hidden_neurons": np.random.randint(50, 200), "activation": np.random.choice(["relu", "tanh"])} for _ in range(pop_size)]
    history = []

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        scores = []
        for individual in population:
            accuracy = evaluate_model(**individual)
            scores.append((individual, accuracy))

        scores.sort(key=lambda x: x[1], reverse=True)
        history.extend(scores)

        parents = [x[0] for x in scores[:pop_size // 2]]
        next_population = parents.copy()

        for _ in range(pop_size - len(parents)):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = {
                "hidden_neurons": np.random.choice([parent1["hidden_neurons"], parent2["hidden_neurons"]]),
                "activation": np.random.choice([parent1["activation"], parent2["activation"]])
            }
            if np.random.rand() < mutation_rate:
                child["hidden_neurons"] = np.random.randint(50, 200)
            next_population.append(child)

        population = next_population

    with open("results/evolutionary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Hidden Neurons", "Activation", "Accuracy"])
        for generation, (individual, acc) in enumerate(history, start=1):
            writer.writerow([generation, individual["hidden_neurons"], individual["activation"], acc])

    return scores[0]


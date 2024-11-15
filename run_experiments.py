from evolutionary.genetic_algorithm import genetic_algorithm
from random_search.random_search import random_search

if __name__ == "__main__":
    print("Running Genetic Algorithm...")
    best_genetic = genetic_algorithm()

    print("Running Random Search...")
    best_random = random_search()

    print("\nBest Genetic Algorithm Result:")
    print(best_genetic)

    print("\nBest Random Search Result:")
    print(best_random)

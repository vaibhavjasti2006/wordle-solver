import numpy as np
from wordle_solver import WordEntropyGraph

# Read both files and add words to lists:
with open('data/valid_guesses.txt', 'r', encoding='utf-8') as file:
    guesses = file.read().split()

with open('data/valid_answers.txt', 'r', encoding='utf-8') as file:
    answers = file.read().split()

# Convert to np.ndarray:
guesses = np.array(guesses)
answers = np.array(answers)

# Build the Word graph:
graph = WordEntropyGraph(guesses, answers)

# Save the graph:
graph.save_graph("data/graph.npy")
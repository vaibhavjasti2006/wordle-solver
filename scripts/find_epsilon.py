import numpy as np
from tqdm import tqdm
from wordle_solver import WordEntropyGraph, pattern

# Epsilon values to look over and maximum number of turns a try is allowed:
MAX_TURNS = 20
EPSILON_VALUES = [round(0.01 * i, 2) for i in range(1, 11)]

# Method to parse a file and set up the word array:
def load_word_list(path):
    with open(path) as f:
        return np.array([w.upper() for w in f.read().split()])

# To play against one word:
def play_one_game(graph, true_answer):
    graph.reset()
    for turn in range(1, MAX_TURNS + 1):
        guess = graph.get_ranked_candidates()[0][0]
        code = pattern.compute_pattern(guess, true_answer)
        feedback = pattern.decode(code)
        graph.update_beliefs(guess, feedback)
        if feedback == "GGGGG":
            return turn
    return MAX_TURNS

# Average num turns needed for an epsilon value:
def average_for_epsilon(guesses, answers, epsilon, all_answers):
    graph = WordEntropyGraph(guesses, answers, file_path="data/graph.npy", epsilon=epsilon)
    counts = []
    for answer in tqdm(all_answers, desc=f"epsilon={epsilon}"):
        counts.append(play_one_game(graph, answer))
    return sum(counts) / len(counts)

# Main block setup to run against all epsilon values:
def main():
    guesses = load_word_list("data/valid_guesses.txt")
    answers = load_word_list("data/valid_answers.txt")

    results = []
    for eps in EPSILON_VALUES:
        avg = average_for_epsilon(guesses, answers, eps, answers)
        print(f"epsilon={eps:<6} avg={avg:.4f}", flush=True)
        results.append((eps, avg))
        with open("data/epsilon_sweep_results.txt", "a") as f:
            f.write(f"{eps},{avg:.4f}\n")

    best_eps, best_avg = min(results, key=lambda x: x[1])
    print(f"\nBest: epsilon={best_eps} with avg={best_avg:.4f}")

if __name__ == "__main__":
    main()
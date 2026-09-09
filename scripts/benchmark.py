import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from wordle_solver import WordEntropyGraph, pattern

# Maximum number of turns a try is allowed:
MAX_TURNS = 20

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

# Runs play_one_game against every given answer, returns a dict of
# word -> number of guesses it took:
def run_benchmark(graph, test_answers):
    results = {}
    for answer in tqdm(test_answers, desc="Benchmarking"):
        results[answer] = play_one_game(graph, answer)
    return results

# Prints average and worst-case guess counts:
def summarize(results):
    counts = list(results.values())
    average = sum(counts) / len(counts)
    worst = max(counts)
    worst_words = [w for w, c in results.items() if c == worst]
    print(f"Tested on {len(counts)} words")
    print(f"Average guesses: {average:.3f}")
    print(f"Worst case: {worst} guesses ({', '.join(worst_words)})")
    return counts

# Bar chart: how many words were solved in each number of guesses:
def plot_distribution(counts):
    tally = Counter(counts)
    turn_numbers = sorted(tally)
    word_counts = [tally[t] for t in turn_numbers]

    plt.bar(turn_numbers, word_counts)
    plt.xlabel("Guesses to solve")
    plt.ylabel("Number of words")
    plt.title(f"Distribution of guesses to solve (n={sum(word_counts)})")
    plt.savefig("benchmark_distribution.png")
    print("Saved chart to benchmark_distribution.png")

# Main block setup to run the benchmark at the empirically best epsilon:
def main():
    guesses = load_word_list("data/valid_guesses.txt")
    answers = load_word_list("data/valid_answers.txt")

    graph = WordEntropyGraph(guesses, answers, file_path="data/graph.npy", epsilon=0.08)

    results = run_benchmark(graph, answers)
    counts = summarize(results)
    plot_distribution(counts)

if __name__ == "__main__":
    main()
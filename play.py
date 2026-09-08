import numpy as np
from wordle_solver import WordEntropyGraph

# Load word lists and build/load the graph once at startup:
with open("data/valid_guesses.txt") as f:
    guesses = np.array(f.read().split())
with open("data/valid_answers.txt") as f:
    answers = np.array(f.read().split())

graph = WordEntropyGraph(guesses, answers, file_path="data/graph.npy")

def print_candidates():
    # Prints the current top 10 ranked guesses, their expected info gain,
    # and whether each is still a possible answer:
    print("\nTop 10 candidates:")
    for word, entropy in graph.get_ranked_candidates()[:10]:
        possible = "possible" if not graph.is_eliminated(word) else "eliminated"
        print(f"  {word}   {entropy:.3f}   {possible}")
    print()

def get_word():
    # Repeatedly prompts until a valid 5-letter, alphabetic, real guess
    # word is entered -- covers every invalid case separately with its
    # own message:
    while True:
        word = input("word: ").strip().upper()
        if len(word) != 5:
            print("Invalid word: must be exactly 5 letters")
            continue
        if not word.isalpha():
            print("Invalid word: letters only")
            continue
        if word not in graph.guess_index:
            print("Invalid word: not in guess list")
            continue
        return word

def get_feedback():
    # Repeatedly prompts until a valid 5-character G/Y/B feedback string
    # is entered:
    while True:
        fb = input("feedback (G/Y/B): ").strip().upper()
        if len(fb) != 5:
            print("Invalid feedback: must be exactly 5 characters")
            continue
        if not all(c in ("G", "Y", "B") for c in fb):
            print("Invalid feedback: use only G, Y, or B")
            continue
        return fb

def main():
    print("Wordle Solver")
    print_candidates()

    while True:
        word = get_word()
        pattern = get_feedback()

        info_gained = graph.get_information_gain(word, pattern)
        graph.update_beliefs(word, pattern)

        print(f"\nInfo gained: {info_gained:.3f} bits")

        if pattern == "GGGGG":
            print("\nWordle solved!")
            break

        print_candidates()

if __name__ == "__main__":
    main()
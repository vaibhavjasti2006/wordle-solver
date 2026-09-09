import numpy as np
from wordle_solver.word_loader import WordLoader
from wordle_solver import pattern
from tqdm import tqdm
from functools import cmp_to_key

class WordEntropyGraph:
    """
    Behaviour: Build a guess x answer matrix and tracks which answers are possible candidates given
        feedback so far. The state of the matrix represents a graph.
    Parameters:
        guesses (np.ndarray): Array of all possible guess words.
        answers (np.ndarray): Array of all possible words.
        file_path (str): Optional file path to a precomputed matrix (.npy) to load instead of rebuilding
            from scratch. Assumes the matrix was built from the exact same guesses/answers arrays. Do not use
            if not the case.
        epsilon (float): How much difference the entropy of two words is allowed to be below which they are considered equal.
    """
    def __init__(self, guesses: np.ndarray, answers: np.ndarray, file_path: str=None, epsilon=0.01):
            self.epsilon = epsilon
            
            # Build the WordLoader with the given guesses and answers:
            loader = WordLoader(guesses, answers)
            self.guess_index = loader.get_guess_index()
            self.answer_index = loader.get_answer_index()
    
            # Load from file if available else compute the matrix from scratch:
            if file_path is not None:
                self.graph = np.load(file_path)
            else:
                self.graph = loader.build_matrix()
                for guess, row in tqdm(self.guess_index.items(), desc="Building pattern matrix"):
                    for answer, col in self.answer_index.items():
                        self.graph[row, col] = pattern.compute_pattern(guess, answer)
    
            # Build a mask with the initial setting to consider all answers:
            self.mask = np.full(self.graph.shape[1], True)

    def update_beliefs(self, guess: str, feedback_pattern: str) -> set[str]:
        """
        Behaviour: Given a guess and the feedback received for it, eliminates
            any remaining candidate answers inconsistent with that feedback.
        Returns: The elimated words as a set of strings.
        Parameters:
            guess (str): The guess word.
            feedback_pattern (str): The feedback pattern string recieved.
        """
        eliminated = set()
        code = pattern.encode(feedback_pattern)
        guess_idx = self.guess_index[guess]

        # Moves through the possible answers words and eliminates the words that do not
        # match the pattern:
        for answer, idx in self.answer_index.items():
            if self.mask[idx] and not self.graph[guess_idx, idx] == code:
                self.mask[idx] = False
                eliminated.add(answer)

        return eliminated

    def get_ranked_candidates(self) -> list[tuple[str, float]]:
        """
        Behaviour: Computes the expected information gain (entropy) of every
            valid guess against the currently remaining candidate answers,
            and ranks guesses from highest to lowest expected gain.
        Returns: A list of (word, expected_entropy) tuples, sorted descending
            by expected_entropy.
        """
        remaining = self.graph[:, self.mask]
        total = remaining.shape[1]
        results = []

        # Calculates the expected entropy for each guess word:
        for guess, row in self.guess_index.items():
            codes = remaining[row]
            _, counts = np.unique(codes, return_counts=True)
            prob_vec = counts / total
            exp_entropy = -np.sum(prob_vec * np.log2(prob_vec))
            results.append((guess, exp_entropy))

        # Custom comparision function:
        # Sort by entropy if difference is > 0.01 else by wether or not smtg is eliminated
        def compare(a, b):
            if abs(a[1] - b[1]) > self.epsilon:
                return -1 if a[1] > b[1] else 1
            a_possible = not self.is_eliminated(a[0])
            b_possible = not self.is_eliminated(b[0])
            if a_possible == b_possible:
                return 0
            return -1 if a_possible else 1

        # Sorts the guess words by entropy and then by valid guesses or not:
        results.sort(key=cmp_to_key(compare))
        return results

    def get_information_gain(self, guess: str, feedback_pattern: str) -> float:
        """
        Behaviour: Computes realized information gain (-log2(p)) for a guess
            given actual feedback.
        Returns: Information gain in bits. inf if the feedback is impossible
            given the current candidate pool.
        Parameters:
            guess (str): The word that was guessed.
            feedback_pattern (str): The letter-based feedback (ex: GGYBB).
        """
        # The integer code for the pattern:
        code = pattern.encode(feedback_pattern)
        guess_idx = self.guess_index[guess]

        # Counts the remaining words:
        remaining = self.graph[guess_idx, self.mask]
        total = remaining.shape[0]
        count = np.sum(remaining == code)

        # Returns the information gain in bits:
        p = count / total
        return -np.log2(p)

    def is_eliminated(self, word: str) -> bool:
        """
        Behaviour: Checks whether a word can still be the answer.
        Returns: True if the word is not a live candidate, False otherwise.
        Parameters:
            word (str): The word to check.
        """
        idx = self.answer_index.get(word, None)
        return idx is None or not self.mask[idx]

    def save_graph(self, file_path: str) -> None:
        """
        Behaviour: Saves the current pattern matrix to disk so it can be
            reloaded later without rebuilding.
        Parameters:
            file_path (str): Destination path for the saved matrix (.npy).
        """
        np.save(file_path, self.graph)

    def reset(self):
        """
        Behaviour: Resets the candidate mask so every answer is considered
            possible again, without rebuilding the underlying matrix.
        """
        self.mask[:] = True
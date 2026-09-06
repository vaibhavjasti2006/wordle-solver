import numpy as np

WORD_LENGTH = 5

class WordLoader:
    """
    Behaviour: Takes raw guesses and answers as word lists.
        Processes them by making sure each word is the appropriate length (WORD_LENGTH=5),
        is built only from alphabets, contains no duplicates and converts them to upper case.
    Parameters:
        guesses (np.ndarray): Array of all possible guess words.
        answers (np.ndarray): Array of all possible answers.
    """
    def __init__(self, guesses: np.ndarray, answers: np.ndarray):
        # Make duplicates for the input arrays:
        guesses = guesses.copy()
        answers = answers.copy()

        # Remove all words whose length is not WORD_LENGTH:
        mask = np.vectorize(lambda s: len(s) == WORD_LENGTH and s.isalpha())
        guesses = guesses[mask(guesses)]
        answers = answers[mask(answers)]

        # Upper Case everywhere:
        guesses = np.char.upper(guesses)
        answers = np.char.upper(answers)

        # Remove all duplicates:
        guesses = np.unique(guesses)
        answers = np.unique(answers)

        # Build the mappings between word and index for the guesses and answers:
        self.guess_index = {word : index for index, word in enumerate(guesses)}
        self.answer_index = {word : index for index, word in enumerate(answers)}

    def build_matrix(self) -> np.ndarray:
        """
        Behaviour: Builds a guess x answer matrix representing a graph.
        Returns: A guess x matrix where each row corresponds to a valid guess and each
            column corresponds to a valid answer. The matrix can only store values of
            type np.uint8. The matrix is filled with a dummy value of 255.
        """
        num_rows = len(self.guess_index)
        num_cols = len(self.answer_index)
        return np.full((num_rows, num_cols), 255, np.uint8)

    def get_guess_index(self) -> dict:
        """
        Returns: A dict mapping guesses to their indices. This convention should
            be maintained when using the build_matrix method.
        """
        return self.guess_index

    def get_answer_index(self) -> dict:
        """
        Returns: A dict mapping answers to their indices. This convention should
            be maintained when using the build_matrix method.
        """
        return self.answer_index
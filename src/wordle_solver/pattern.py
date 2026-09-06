import numpy as np
from collections import Counter

GREEN, YELLOW, BLACK = "0", "1", "2"
WORD_LENGTH = 5

def encode(pattern: str) -> int:
    """
    Behaviour: Converts a letter based pattern string (using "G", "Y", "B") to its base-3 encoding.
    Returns: The integer encoding (0-242) of the given pattern.
    Parameters:
        pattern (str): The string pattern (ex: GGYBB) to be turned into an encoded integer.
            Assumes that the string is WORD_LENGTH=5
    """
    codes = {"G": GREEN, "Y": YELLOW, "B": BLACK}
    pattern = [codes[letter] for letter in list(pattern)]
    return int("".join(pattern), 3)

def decode(code: int) -> str:
    """
    Behaviour: Converts a base-3 number (0-242) integer encoding back to its letter based
        pattern string. Inverse of encode.
    Returns: The letter based pattern string (ex: GGYBB)
    Parameters:
        code (int): The integer encoding to be converted to a pattern string.
            Assume the code is between (0-242).
    """
    base3_repn = list(np.base_repr(code, base=3).zfill(WORD_LENGTH))
    codes = {GREEN: "G", YELLOW: "Y", BLACK: "B"}
    pattern = [codes[digit] for digit in base3_repn]
    return "".join(pattern)

def compute_pattern(guess: str, answer: str) -> int:
    """
    Behaviour: Given a guess and an answer, it computes the correct corresponding encoded wordle pattern.
    Returns: The wordle pattern as an integer between (0-242).
    Parameters:
        guess (str): A possible guess word.
        answer (str): A possible answer, the guess word is mapped against.
    """
    pattern = [""] * WORD_LENGTH
    counter = Counter(answer)

    # Move through the letters and decide between green or black:
    for idx, letter in enumerate(guess):
        if letter == answer[idx]:
            pattern[idx] = GREEN
            counter[letter] -= 1
        else:
            pattern[idx] = BLACK

    # For all the letters that were previously marked black, check if they could be yellow:
    for idx, letter in enumerate(guess):
        if pattern[idx] == BLACK:
            if letter in counter and counter[letter] > 0:
                counter[letter] -= 1
                pattern[idx] = YELLOW

    return int("".join(pattern), 3)
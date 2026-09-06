import numpy as np
from wordle_solver import WordLoader

def test_filter_wrong_length():
    # Construct the WordLoader object:
    guesses = np.array(["CRANE", "TOAST", "BLUE", "HUH"])
    answers = np.array(["CRANE", "TWAT", "TOAST"])
    loader = WordLoader(guesses, answers)

    # Checks for the guesses:
    guess_index = loader.get_guess_index()
    assert "BLUE" not in guess_index
    assert "HUH" not in guess_index
    assert len(guess_index) == 2

    # Checks for the answers:
    answer_index = loader.get_answer_index()
    assert "TWAT" not in answer_index
    assert len(answer_index) == 2

def test_filters_not_alpha():
    # Construct the WordLoader object:
    guesses = np.array(["CRANE", "CR4NE", "TO@ST"])
    answers = np.array(["CRANE", "CR4NE"])
    loader = WordLoader(guesses, answers)

    # Check the guesses:
    guess_index = loader.get_guess_index()
    assert "CR4NE" not in guess_index
    assert "TO@ST" not in guess_index
    assert "CRANE" in guess_index
    assert len(guess_index) == 1

    # Check the answers:
    answer_index = loader.get_answer_index()
    assert "CR4NE" not in answer_index
    assert len(answer_index) == 1

def test_case_normalization_and_no_dups():
    # Construct the WordLoader object:
    guesses = np.array(["crane", "CRANE", "Crane"])
    answers = np.array(["crane", "CRANE"])
    loader = WordLoader(guesses, answers)

    # Check the guesses:
    guess_index = loader.get_guess_index()
    assert "CRANE" in guess_index
    assert len(guess_index) == 1

    # Check the answers:
    answer_index = loader.get_answer_index()
    assert "CRANE" in answer_index
    assert len(answer_index) == 1

def test_build_matrix_shape_and_fill():
    # Construct the WordLoader object:
    guesses = np.array(["CRANE", "TOAST", "STARE", "B456"])
    answers = np.array(["CRANE", "TOAST", "HMM"])
    loader = WordLoader(guesses, answers)

    # Check the matrix:
    matrix = loader.build_matrix()
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.uint8
    assert (matrix == 255).all()
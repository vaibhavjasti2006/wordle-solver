import numpy as np
from wordle_solver import WordEntropyGraph, compute_pattern

def make_graph():
    # Method to return a built WordEntropyGraph:
    guesses = np.array(["CRANE", "TOAST", "STARE"])
    answers = np.array(["CRANE", "TOAST"])
    return WordEntropyGraph(guesses, answers)

def test_matrix_cells_match_compute_pattern():
    # Confirms row/col indexing lines up correctly, each cell should
    # equal compute_pattern(guess, answer) for that exact pair:
    graph = make_graph()
    for guess, row in graph.guess_index.items():
        for answer, col in graph.answer_index.items():
            assert graph.graph[row, col] == compute_pattern(guess, answer)

def test_update_beliefs_narrows_mask():
    graph = make_graph()
    # Guessing the actual answer CRANE against itself gives all green.
    # Only CRANE itself should remain a candidate afterward:
    graph.update_beliefs("CRANE", "GGGGG")

    assert not graph.is_eliminated("CRANE")
    assert graph.is_eliminated("TOAST")

def test_update_beliefs_returns_eliminated_set():
    # Check if words are elimnated appropriately:
    graph = make_graph()
    eliminated = graph.update_beliefs("CRANE", "GGGGG")
    assert eliminated == {"TOAST"}


def test_update_beliefs_ignores_already_eliminated():
    graph = make_graph()
    graph.update_beliefs("CRANE", "GGGGG")
    # TOAST is already eliminated, a second call shouldn't re-add it:
    eliminated_again = graph.update_beliefs("TOAST", "BBBBB")
    assert "TOAST" not in eliminated_again

def test_get_ranked_candidates():
    # Checks if the ranked candidates are ranked correctly:
    graph = make_graph()
    ranked = graph.get_ranked_candidates()
    entropies = [entropy for _, entropy in ranked]
    assert entropies == sorted(entropies, reverse=True)

def test_is_eliminated_for_guess_only_word():
    # STARE is a valid guess but not a possible answer in this tiny setup,
    # should be treated as eliminated immediately:
    graph = make_graph()
    assert graph.is_eliminated("STARE")

def test_reset_restores_mask():
    # Check if resent sets the graph to initial state:
    graph = make_graph()
    graph.update_beliefs("CRANE", "GGGGG")
    assert graph.is_eliminated("TOAST")

    graph.reset()
    assert not graph.is_eliminated("TOAST")

def test_save_and_load_matrix(tmp_path):
    # Checks if the saving and reloading works correctly:
    graph = make_graph()
    file_path = tmp_path / "test_matrix.npy"
    graph.save_graph(str(file_path))

    guesses = np.array(["CRANE", "TOAST", "STARE"])
    answers = np.array(["CRANE", "TOAST"])
    reloaded = WordEntropyGraph(guesses, answers, file_path=str(file_path))

    assert np.array_equal(reloaded.graph, graph.graph)
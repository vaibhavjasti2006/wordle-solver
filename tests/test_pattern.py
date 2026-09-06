import numpy as np
from wordle_solver import compute_pattern, encode, decode

def test_matches():
    # Test the encode method and compute_pattern:
    assert compute_pattern("CRANE", "CRANE") == encode("GGGGG")
    assert compute_pattern("GHOST", "PLUMB") == encode("BBBBB")
    assert compute_pattern("ABCDE", "EABCD") == encode("YYYYY")
    assert compute_pattern("SPEED", "ERASE") == encode("YBYYB")
    assert compute_pattern("SPEED", "STEAK") == encode("GBGBB")

def test_encode_decode():
    # Test if encode and decode act as the inverse of each other:
    for raw in ["GGGGG", "BBBBB", "YYYYY", "GYBGY", "BBBBG"]:
        assert decode(encode(raw)) == raw
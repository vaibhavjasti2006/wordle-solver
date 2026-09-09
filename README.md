# Wordle Solver

An information-theoretic Wordle solver — picks the next guess by maximizing expected information gain (Shannon entropy) over the remaining candidate answers.

## How it works

Every guess partitions the space of possible answers into buckets based on the feedback pattern it would produce (green/yellow/black per letter — 243 possible patterns for a 5-letter word). The solver computes, for every valid guess word, the expected information gain of asking that question and picks the guess that's expected to narrow the candidate pool the most.

Ties in expected entropy are broken by preferring guesses that could still be the actual answer, with a tunable `epsilon` threshold controlling how close two entropy values need to be before they're considered "tied."

## Performance

Benchmarked against the full ~2,315-word answer list, using the empirically tuned `epsilon=0.08`:

- **Average guesses: 3.440**
- **Worst case: 6 guesses** (never exceeds the real Wordle limit)

This solver reaches that range using pure greedy entropy maximization with no lookahead.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs all dependencies and the project itself (in editable mode) into `.venv`.

## Usage

### Play interactively
Run the build_matrix script in scripts before playing. It should take about a minute to finish running.

```bash
uv run python play.py
```

Prompts for a guess, then for the feedback received (as `G`/`Y`/`B` per letter), and shows the top 10 ranked next-best guesses after each turn along with the actual information gained.

### Rebuild the cached pattern matrix

```bash
uv run python scripts/build_matrix.py
```

Precomputes and caches the guess × answer feedback pattern matrix to `data/graph.npy`, so it doesn't need to be rebuilt on every run. Only needs to be rerun if the word lists change.
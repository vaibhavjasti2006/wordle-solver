# Wordle Solver

An information-theoretic Wordle solver — picks the next guess by maximizing expected information gain (Shannon entropy) over the remaining candidate answers.

I built this after getting some inspiration from the 3Blue1Brown video on Wordle, since I've been playing Wordle almost every day over the summer and also listened to a couple of lectures on information theory out of curiosity. I thought this would be a fun way to mix two of my recent interests.

Wordle is a game with a finite number of possibilities, so the entire game tree could in principle be fully mapped out — and has been, reaching a practical optimum of around 3.42 average guesses. This solver takes a greedy, entropy-based approach instead, which turns out to be a very strong heuristic on its own. It's similar to 3Blue1Brown's approach, but with no lookahead — only ever considering one guess into the future.

My initial greedy approach got a meaningful boost from introducing epsilon thresholding: words with very close expected entropy are treated as effectively equal, and ties are then broken by whether or not the word could still be the actual answer. I ran a hyperparameter search over 10 epsilon values to find the best one for this dataset, which gave a noticeable improvement.

You can read some of the stats below, and interact with the bot yourself if you'd like.

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
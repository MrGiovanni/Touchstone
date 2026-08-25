# Touchstone Leaderboard

Machine-readable results for the [Touchstone Benchmark](https://arxiv.org/abs/2411.03670)
(NeurIPS 2024), plus the static site that renders them.

The leaderboard also lives in the repository README as Markdown tables. That
version is for people. This directory is for everything else: agents, scripts,
spreadsheets, and anyone who wants the numbers without 600 characters of badge
markup per row.

## What is here

| Path | What it is |
|---|---|
| `index.html` | The leaderboard as a static page. No dependencies, no build step, works offline. |
| `data/leaderboard.json` | Everything: benchmark facts, model registry, and one record per (model, organ). |
| `data/leaderboard.csv` | The results, flat. One row per (model, organ, metric, value). |
| `data/models.csv` | Per-model metadata: organization, average DSC, parameters, inference speed, paper, code. |
| `build.py` | Regenerates all of the above from the repository README. |
| `SUBMISSION.md` | How to add a model. |

## Benchmark at a glance

Figures below are from the paper and are the authoritative ones.

| | |
|---|---|
| Training | 5,195 CT volumes, 76 hospitals |
| Testing | 5,903 CT volumes, 11 additional hospitals |
| Structures | 9 abdominal organs |
| Algorithms | 19, from 14 research teams |
| Metric | Dice Similarity Coefficient (DSC) |

## Publishing the page

Enable GitHub Pages for this repository with source **Deploy from a branch**,
branch `main`, folder `/ (root)`. The leaderboard is then served at:

```
https://mrgiovanni.github.io/Touchstone/leaderboard/
```

Set that URL as the repository's homepage field.

## Regenerating

The README stays the editable source of truth. After changing its tables:

```bash
cd leaderboard
python build.py ../README.md
```

Then commit `data/` and `index.html`. `build.py` needs only the Python standard
library.

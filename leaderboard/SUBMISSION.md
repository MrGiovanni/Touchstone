# Adding a model to the Touchstone leaderboard

Two tiers. The first you can complete yourself; the second we run.

## Tier 1 — self-reported

For results on the public portion of the test set (TotalSegmentator V2).

1. Train your model on [AbdomenAtlas](https://github.com/MrGiovanni/AbdomenAtlas).
   Do not train on any part of the test set.
2. Predict on the public test volumes and compute per-case DSC and NSD.
3. Open a pull request adding one folder under `totalsegmentator_results/<YourModel>/`
   containing `dsc.csv` and `nsd.csv`, in the same layout as the existing folders.
4. In the same PR, add a row to the README tables and re-run:
   ```bash
   cd leaderboard && python build.py ../README.md
   ```
   Commit the regenerated `data/` and `index.html`.
5. Include in the PR description: model name, organization, paper link, code link,
   parameter count, and the exact training data used.

Self-reported rows are labelled as such on the page.

## Tier 2 — verified

For results on the full held-out test set, including the proprietary JHH cohort.

Open an issue titled `[Submission] <YourModel>` with a link to your checkpoint and
inference script. We run it on the held-out data and report the numbers. Verified
rows carry a badge.

## Metric definition

DSC and NSD implementations disagree, sometimes by tens of points, so the number
only means something if the implementation is pinned. Compute NSD with DeepMind's
[`surface-distance`](https://github.com/google-deepmind/surface-distance) using the
per-organ tolerances stated in the paper. If you use a different implementation,
say so in the PR — we would rather have a labelled difference than a silent one.

## What we ask

- Do not submit a model trained on any portion of the test set.
- State the training data explicitly. "AbdomenAtlas" is fine; "AbdomenAtlas plus
  an internal cohort" needs saying.
- One PR per model.

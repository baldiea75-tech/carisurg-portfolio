## Week 6: Baseline Triage Classification Models

Added this week:

- `notebooks/Week6_Baseline_Models.ipynb`: two baseline classifiers, a logistic regression model and a decision tree, trained on the Week 5 feature set to predict ESI triage level, evaluated against a stratified random baseline.
- `docs/week6_baseline_report.md`: the final report, covering results for both models plus the random baseline, the metric justification for using recall on ESI 1 as the primary metric, and a failure mode reflection.
- `docs/week6_video_script.md`: the script for the 1 minute clinical explainer video.
- `docs/figures/w6_confusion_logreg.png` and `docs/figures/w6_confusion_tree.png`: the confusion matrices for both models.

**Reproducibility.** A fixed random seed of `random_state=42` is used throughout Week 6: for the 80/20 stratified train and test split, for the logistic regression model, and for the decision tree. Re-running `Week6_Baseline_Models.ipynb` end to end on the same data will reproduce the same split and the same results.

**Decision tree depth.** `max_depth` is capped at 5 rather than left unbounded, so the tree stays shallow enough to be explained to a clinician as a short flowchart, at some cost to raw accuracy. See Section 2.1 of `docs/week6_baseline_report.md` for the full justification.


**Last updated:** Saturday 16 July 2026

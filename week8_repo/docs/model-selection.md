# Model Selection: Audit Trail

**Question this answers:** which version gave us 0.74 recall, or any other number, and why did we end up here.

Every model trained across Weeks 6 to 8 is listed below with the same four headline metrics, so any number quoted in a memo, a standup or an interview can be traced back to a specific row here. All models share the same data split: 80 percent train, 20 percent test, stratified on ESI level, random seed 42, confirmed identical across every week by the refactored pipeline in `src/`.

## Full Comparison

| Model | Key Hyperparameters | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Recall (ESI 1) | Train Time | Inference (per patient) | Week |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (stratified random) | none | 0.375 | 0.204 | 0.204 | 0.204 | 0.000 | 0.002s | 0.0003ms | 6 |
| Logistic Regression ★ | max_iter=1000 | 0.667 | 0.582 | 0.463 | 0.492 | 0.250 | 2.7s | 0.001ms | 6 |
| Decision Tree | max_depth=5 | 0.556 | 0.265 | 0.245 | 0.216 | 0.000 | 0.7s | 0.002ms | 6 |
| Random Forest (default) | n_estimators=200, max_depth=10 | 0.580 | 0.387 | 0.261 | 0.238 | 0.000 | 7.7s | 0.017ms | 7 |
| Random Forest (balanced) | n_estimators=200, max_depth=10, class_weight=balanced | 0.478 | 0.395 | 0.492 | 0.369 | 0.250 | 7.5s | 0.018ms | 7 |
| Random Forest (high recall) ★ | n_estimators=150, max_depth=8, class_weight=balanced | 0.470 | 0.392 | 0.549 | 0.367 | 0.562 | 2.7s | 0.007ms | 7 |
| Gradient Boosting (default) | n_estimators=150, learning_rate=0.1, max_depth=3 | 0.664 | 0.520 | 0.415 | 0.446 | 0.062 | 69.4s | 0.014ms | 8 |
| Gradient Boosting (deeper) | n_estimators=150, learning_rate=0.1, max_depth=5 | 0.667 | 0.492 | 0.423 | 0.447 | 0.062 | 120.2s | 0.021ms | 8 |

★ = pinned finalists, frozen in `config.yaml`.

## Reading This Table

**Recall (ESI 1)** is the primary metric for this project, justified in the Week 6 report: a missed ESI 1 patient is a far more serious error than a false alarm on a less urgent case, and ESI 1 makes up roughly 0.1 percent of visits, so accuracy alone hides how a model treats this class entirely.

**Accuracy** is reported alongside it because a model that only chases ESI 1 recall can do so by flagging almost everyone as ESI 1, which is exactly what happens to the random forest variants as class weighting is pushed harder. The two numbers need to be read together, not separately.

## Why Two Models Are Pinned

**Logistic Regression** is the default. It has the strongest overall accuracy of any model tested, is the cheapest and fastest to train and run, and its prediction for a single patient can be explained to a clinician as a short list of weighted factors in under a minute, the bar set in Week 6.

**Random Forest (high recall)** is pinned as a reserve option, not the default. Week 7's wider hyperparameter search found this configuration catches 9 of 16 ESI 1 patients, more than double logistic regression's 4, which may be worth the trade-off in a deployment context where missing a critical patient is judged worse than a high false alarm rate. That trade-off is real and significant: precision on ESI 1 falls to 0.088 (9 correct out of 102 flagged), and overall accuracy drops from 0.667 to 0.470. This model should not be switched to without a specific, documented decision to prioritise sensitivity over overall accuracy.

**Gradient Boosting was tested and not pinned.** Two configurations were tried in Week 8, in response to the question of whether a third model family might close the gap that random forest could not. Neither came close: recall on ESI 1 stayed at 0.062 regardless of tree depth, barely above the level a model predicting almost nothing as ESI 1 would show, while training took 69 to 120 seconds, roughly 25 to 45 times longer than logistic regression for no corresponding benefit. Gradient boosting is kept in this table as part of the audit trail, not because it is a candidate for deployment.

## Reproducibility

Every row above can be regenerated with:

```
python scripts/train.py --config config.yaml --model <model_key>
```

using the model keys in `config.yaml`. Logistic regression and both random forest rows have been directly verified to reproduce these exact figures through the refactored `src/` pipeline, confirmed against the original Week 6 and Week 7 notebooks.

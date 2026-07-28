# Handover Document: CariSurg Triage Model

**The test this document is written to pass:** could someone who has never met me clone this repository, read this one page and be running the model by the end of the day, without asking a single question.

---

## 1. Project Summary

This project predicts Emergency Severity Index (ESI) triage level, from 1 (most critical) to 5 (least critical), from a patient's vital signs and presenting complaints at the moment they arrive in the emergency department. It exists to give clinicians a second opinion at triage, surfacing patients who may need more urgent attention than a first glance suggests, particularly the rare but critical ESI 1 group. It is built for Mercer General Hospital's ED Board and Clinical IT team, and is intended to run alongside, not in place of, a human triage nurse.

## 2. Final Model Decision

**We ship Logistic Regression by default, for its balance of overall accuracy, cost and auditability, and keep a high-recall Random Forest configuration pinned as a reserve option for deployments that prioritise catching ESI 1 patients over overall accuracy.**

Full reasoning is in `docs/decisions/2026-week-7-model-choice.md` and `docs/week7_cost_benefit_memo.md`. The one-line version: no complex model tested in Weeks 7 or 8, including two random forest variants and two gradient boosting variants, beat logistic regression on both accuracy and recall for ESI 1 at the same time. The reserve model exists because more than doubling ESI 1 recall is a real, documented option, not because it is a better model outright.

## 3. How to Run

```
git clone https://github.com/baldiea75-tech/carisurg-portfolio.git
cd carisurg-portfolio
pip install -r requirements.txt
python scripts/train.py --config config.yaml
```

This trains and evaluates the default model (logistic regression) and prints accuracy, precision, recall and F1, per class and overall. To run the reserve model instead:

```
python scripts/train.py --config config.yaml --model random_forest_high_recall
```

Before either command, place the raw dataset at `data/yaleemmlc_admissionprediction_triage.csv` (see Section 4). Run `pytest tests/` first if you want to confirm your environment is set up correctly; both tests should pass in under 10 seconds.

## 4. Where the Data Lives

The raw dataset, `yaleemmlc_admissionprediction_triage.csv`, is **not** committed to this repository. It is listed in `.gitignore`, both because of its size (approximately 55 MB) and because it is clinical data. De-identified does not mean ungoverned: it must be requested through the Mercer Research Ethics Committee process described in `data/README.md`, kept only in the local `data/` folder or an approved secure Drive location and must not be redistributed, screenshotted into chat tools or pasted into any AI tool beyond summary statistics and column names.

## 5. Known Limitations

- **Single-site data.** Every number in this project comes from one US academic hospital (Yale New Haven). Distribution shift is likely at Mercer General, a Caribbean ED with a different patient mix, and no result here should be treated as validated locally until that comparison is done.
- **ESI 1 recall is still modest, even at its best.** The pinned reserve model catches around half of the most critical patients, at the cost of a high false alarm rate. Neither pinned model is accurate enough to run without a clinician reviewing every case; this is a decision support tool, not a replacement for triage judgement.
- **Demographic features are excluded by design, not by oversight.** Age, race, ethnicity, insurance status and similar columns are deliberately left out of the feature set, so the model cannot use a protected characteristic as a shortcut to a prediction. This is a fairness safeguard, but it also means the model may be missing genuine clinical signal that a human triage nurse would otherwise weigh; this trade-off has not been quantified.

## 6. Who to Ask

| Topic | Contact |
|---|---|
| Model and pipeline questions | Ashi Baldie, project author |
| Clinical validity of model output | Dr. Marcus Reyes, Consultant Emergency Physician |
| Data access and governance | Martina Griffith, Clinical IT Lead |
| Project sponsor and scope | Dr. De Fretias |

---

*This document seeds the top-level `README.md`. If the two drift apart, this file is the source of truth for anyone running the model for the first time.*

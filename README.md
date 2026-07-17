## Week 5: Data Exploration and Feasibility (Interim)

Added this week:

- `notebooks/Week5_Interim_Data_Exploration.ipynb`: full profiling of the Yale EMMLC triage dataset (55,121 encounters, 225 columns), covering `.info()`, `.describe()`, a dtype audit, missingness reported as both raw counts and percentages, an outlier report, a data quality issues table, the four plot data quality dashboard and ESI correlation analysis.
- `docs/week5_feasibility_memo.md`: the feasibility memo for the ED Board, covering the verdict, dataset summary, top 3 concerns, top 3 reasons to proceed, caveats and the top 10 feature shortlist for Week 6 modelling.
- `docs/figures/`: the missingness heatmap and the four dashboard plots referenced in the memo, committed as `.png` files.

**Note on the raw data.** `yaleemmlc_admissionprediction_triage.csv` is not committed to this repository. It is de-identified clinical data and is large, roughly 55 MB, so it is listed in `.gitignore` and should be kept in your local or Drive copy only.



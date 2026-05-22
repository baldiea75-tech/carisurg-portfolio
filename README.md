# CariSurg MedTech Pathways — Week 0 Portfolio

**Programme:** CariSurg MedTech Pathways 2026  
**Author:** Ashi Baldie

**Dataset:** Emergency Triage Dataset — 2,205 patients · 11 columns

---

## Table of Contents

- [About This Project](#about-this-project)
- [Dataset Overview](#dataset-overview)
- [Weekly Progress](#weekly-progress)
  - [Day 1 — Gender Column Cleaning](#day-1--gender-column-cleaning)
  - [Day 2 — Pulse Column Cleaning](#day-2--pulse-column-cleaning)
  - [Day 3 — Data Visualisation](#day-3--data-visualisation)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Programme Info](#programme-info)

---

## About This Project

This repository documents my Week 0 work for the CariSurg MedTech Pathways Programme. The goal is to perform exploratory data analysis (EDA) and clinical data cleaning on a dirty emergency triage dataset, identify at-risk patients using rule-based logic, and communicate findings clearly.

The dataset simulates real triage records from Mercer General Hospital's Emergency Department, with intentionally introduced data quality issues reflecting what real clinical data looks like in practice.

---

## Dataset Overview

**File:** `EmergencyTriageDataset_Reduced_Dirty.csv`  
**Rows:** 2,205 patients  
**Columns:** 11

| Column | Raw Dtype | Description | Valid Range |
|--------|-----------|-------------|-------------|
| `ID` | int | Patient identifier | — |
| `Age` | int | Patient age (years) | 18–98 in dataset |
| `Gender` | object | Patient sex | 0 = Female, 1 = Male |
| `GCS` | object | Glasgow Coma Scale — consciousness | 3–15 |
| `SBP` | object | Systolic Blood Pressure (mmHg) | 50–250 |
| `DBP` | float | Diastolic Blood Pressure (mmHg) | 30–150 |
| `MAP` | float | Mean Arterial Pressure (mmHg) | 40–180 |
| `pulse` | object | Heart rate (bpm) | 20–250 |
| `Temp` | object | Body temperature (°C) | 32–43 |
| `RR` | float | Respiratory Rate (breaths/min) | 5–60 |
| `Fio2` | float | Fraction of Inspired Oxygen (%) | 21–100 |

> Columns stored as text (`object`) required type conversion before any numeric analysis was possible.

---

## Weekly Progress

---

### Day 1 — Gender Column Cleaning

**Date:** Wednesday  
**Status:** Complete  
**Google Colab:** [`week0_triage_eda.ipynb`](week0_triage_eda.ipynb)

#### What I found

The `Gender` column contained **6 variants** of two values, caused by inconsistent data entry (mixed case, numeric codes):

| Raw value | Count | Meaning |
|-----------|-------|---------|
| `1` | 422 | Male |
| `MALE` | 379 | Male |
| `Male` | 375 | Male |
| `FEMALE` | 366 | Female |
| `Female` | 340 | Female |
| `0` | 323 | Female |

#### Approach

Built a mapping dictionary standardising all 6 variants to binary integers (`1 = Male`, `0 = Female`), consistent with the Harvard dataset convention used in this programme.

Used `.map()` instead of `.replace()` so that any unrecognised value would surface as `NaN` rather than being silently skipped — acting as a validation mechanism. Added an assertion to confirm only `{0, 1}` exist after cleaning.

#### Improvements applied (from feedback)

- Added inline comments explaining each decision
- Wrapped cleaning logic in a reusable `clean_gender()` function
- Documented assumptions about allowed variants directly in code
- Added post-cleaning assertion to validate the result fails loudly if unexpected values appear

#### Result

| Class | Count |
|-------|-------|
| Male (1) | 1,176 |
| Female (0) | 1,029 |
| Missing (NaN) | 0 |

#### Key code

```python
def clean_gender(series):
    """
    Standardise Gender column to binary integer encoding.
    Convention: 1 = Male, 0 = Female (matches Harvard dataset standard).
    Any unrecognised value becomes NaN — surfaced deliberately so issues
    are caught early rather than silently passed through.
    """
    gender_map = {
        'Male': 1, 'MALE': 1, '1': 1,    # all Male variants
        'Female': 0, 'FEMALE': 0, '0': 0  # all Female variants
    }
    return series.map(gender_map)  # unmatched → NaN (intentional)

df['Gender'] = clean_gender(df['Gender'])

# Validate — fails loudly if unexpected categories appear
assert set(df['Gender'].dropna().unique()) == {0, 1}, \
    f"Unexpected Gender values found: {df['Gender'].unique()}"
assert df['Gender'].isnull().sum() == 0, "Unexpected NaNs in Gender after cleaning"
```

---

### Day 2 — Pulse Column Cleaning

**Date:** Thursday  
**Status:** Complete  
**Google Colab:** [`week0_triage_eda_day2.ipynb`](week0_triage_eda_day2.ipynb)

#### What I found

The `pulse` (heart rate) column was stored as `object` (string) dtype despite being a numeric measurement. After `pd.to_numeric()` conversion:

- **44 NaNs** created from non-numeric text entries
- **43 additional rows** had physiologically impossible values: `0.0` (impossible for a living patient) and `300.0` (far above any survivable heart rate)
- Valid range used: **20–250 bpm** (per CariSurg clinical reference table)

#### Approach

1. Converted to numeric with `errors='coerce'` — non-numeric text → NaN
2. Flagged and replaced physiologically impossible values (< 20 or > 250) with NaN
3. Imputed all NaNs using the **median (90.0 bpm)** — chosen over mean because heart rate data is right-skewed with outliers
4. Companion-cleaned `SBP` in the same step to enable the composite shock index flag
5. Wrapped each stage in a reusable function with inline clinical justification

#### Clinical significance

Heart rate is critical for detecting:

- **Tachycardia (> 100 bpm)** — fever, pain, blood loss, anxiety
- **Shock index** — when combined with SBP < 90: indicates haemodynamic compromise
- **Bradycardia (< 60 bpm)** — heart block, medication effect

#### Composite Shock Flag

| SBP | Pulse | Interpretation |
|-----|-------|----------------|
| < 90 mmHg | > 110 bpm | Suspected decompensated shock |

**Result: 32 patients (1.45%) flagged as potential shock cases.**

#### Result after cleaning

| Metric | Value |
|--------|-------|
| Min | 40.0 bpm |
| Median | 90.0 bpm |
| Max | 170.0 bpm |
| NaNs remaining | 0 |

#### Visualisation

![Pulse Analytics](pulse_analytics.png)

---

### Day 3 — Data Visualisation

**Date:** Friday  
**Status:** Complete  
**Google Colab:** [`week0_triage_eda_day3.ipynb`](week0_triage_eda_day3.ipynb)

#### Overview

Produced six clinically meaningful plots from the fully cleaned triage dataset. Every plot answers a specific clinical question. Cleaning from Days 1 and 2 was consolidated into a single reusable `apply_all_cleaning()` function and called once before any plotting. A seventh custom plot was added as the open-ended exercise.

#### Plots produced

| File | Plot type | Clinical question answered |
|------|-----------|---------------------------|
| `gender_distribution.png` | Bar chart | What is the sex breakdown of triage patients? |
| `gcs_histogram.png` | Histogram | Are most patients fully alert, or many presenting with reduced consciousness? |
| `pulse_histogram.png` | Histogram | What proportion present with abnormal heart rates? |
| `sbp_vs_dbp.png` | Scatter | Does the SBP/DBP relationship follow expected physiology? |
| `age_vs_pulse.png` | Scatter | Does heart rate vary with patient age? |
| `vitals_boxplots.png` | Box plots (multi-panel) | Do all vital sign distributions sit within plausible ranges post-cleaning? |
| `Respiratory Rate vs Temperature.png` | Scatter (custom) | Do fever and tachypnoea co-occur, and does the pattern suggest sepsis or pneumonia? |

#### Custom plot - Temperature vs Respiratory Rate

**Clinical question:** Do fever and tachypnoea co-occur, and is this pattern more consistent with **sepsis** or **pneumonia**?

Both fever (Temp > 38.0 °C) and tachypnoea (RR > 20 /min) are SIRS criteria — Systemic Inflammatory Response Syndrome. When a patient meets 3 or more SIRS criteria the working clinical diagnosis shifts toward sepsis, not a localised infection like pneumonia. Pneumonia would additionally require raised FiO2 (supplemental oxygen), indicating respiratory failure at the organ level.

The data was checked against both patterns:

| Pattern | Criteria | Patients flagged |
|---------|----------|-----------------|
| Dual SIRS flag | Fever + tachypnoea | 133 (6.0%) |
| SIRS ≥ 3 criteria | Fever + tachypnoea + HR > 100 | 88 (4.0%) |
| Pneumonia-like | Fever + tachypnoea + FiO2 > 21% | 29 (1.3%) |

Points are colour-coded by clinical zone:

- **Pink** - fever + tachypnoea
- **Yellow** - fever only
- **Grey** - no fever

Reference lines mark the SIRS fever threshold (38.0 °C) and tachypnoea threshold (20 /min), dividing the chart into four clinical quadrants. The high-risk quadrant is annotated with both the dual-flag count.

![Temperature vs Respiratory Rate](RRvsTemperature.png)

---

## Repository Structure

```
carisurg-portfolio/
│
├── README.md                                     ← this file
├── week0_triage_eda.ipynb                        ← Day 1 notebook
├── week0_triage_eda_day2.ipynb                   ← Day 2 notebook
├── week0_triage_eda_day3.ipynb                   ← Day 3 notebook
├── EmergencyTriageDataset_Reduced_Dirty.csv      ← raw dataset
│
└── plots/
    ├── gender_distribution.png                   ← Day 3
    ├── gcs_histogram.png                         ← Day 3
    ├── pulse_histogram.png                       ← Day 3
    ├── sbp_vs_dbp.png                            ← Day 3
    ├── age_vs_pulse.png                          ← Day 3
    ├── vitals_boxplots.png                       ← Day 3
    ├── my_plot.png                               ← Day 3 custom plot
    └── pulse_analytics.png                       ← Day 2
```

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `EmergencyTriageDataset_Reduced_Dirty.csv` to your Google Drive
3. Open the relevant notebook for the day you want to run
4. Select **Runtime → Restart and run all**
5. The file auto-locates from Drive — no manual path editing needed

**Dependencies** (all pre-installed in Colab):  
`pandas` · `numpy` · `matplotlib`

---

## Programme Info

**Programme:** CariSurg MedTech Pathways 2026

---

*Last updated: Day 3 - Friday*

# CariSurg MedTech Pathways — Week 0 Portfolio

**Programme:** CariSurg MedTech Pathways 2026  
**Author:** Ashi Baldie

**Dataset:** Emergency Triage Dataset - 2,205 patients · 11 columns  

---

## Table of Contents

- [About This Project](#about-this-project)
- [Dataset Overview](#dataset-overview)
- [Weekly Progress](#weekly-progress)
  - [Day 1 — Gender Column Cleaning](#day-1--gender-column-cleaning)
  - [Day 2 — Pulse Column Cleaning](#day-2--pulse-column-cleaning)
  
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
| `Temp` | object  | Body temperature (°C) | 32–43 |
| `RR` | float | Respiratory Rate (breaths/min) | 5–60 |
| `Fio2` | float | Fraction of Inspired Oxygen (%) | 21–100 |

>  Columns stored as text (`object`) required type conversion before any numeric analysis was possible.

---

## Weekly Progress

---

### Day 1 — Gender Column Cleaning

**Date:** Wednesday  
**Status:** Complete  
**Notebook:** [`week0_triage_eda.ipynb`](week0_triage_eda.ipynb)

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
        'Male': 1, 'MALE': 1, '1': 1,   # all Male variants
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
**Notebook:** [`week0_triage_eda.ipynb`](week0_triage_eda.ipynb)

#### What I found

The `pulse` (heart rate) column was stored as `object` (string) dtype despite being a numeric measurement. After `pd.to_numeric()` conversion:

- **44 NaNs** created from non-numeric text entries
- **43 additional rows** had physiologically impossible values: `0.0` (impossible for a living patient) and `300.0` (far above any survivable heart rate)
- Valid range used: **20–250 bpm** (per CariSurg clinical reference table)

#### Approach

1. Converted to numeric with `errors='coerce'` — non-numeric text → NaN
2. Flagged and replaced physiologically impossible values (< 20 or > 250) with NaN
3. Imputed all NaNs using the **median** (90.0 bpm) - chosen over mean because heart rate data is right-skewed with outliers
4. Companion-cleaned `SBP` in the same step to enable the composite shock index flag
5. Wrapped each stage in a reusable function with inline clinical justification

#### Clinical significance

Heart rate is critical for detecting:
- **Tachycardia** (> 100 bpm) — fever, pain, blood loss, anxiety
- **Shock index** — when combined with SBP < 90: indicates haemodynamic compromise
- **Bradycardia** (< 60 bpm) — heart block, medication effect

#### Composite Shock Flag (advanced)

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

![Pulse Analytics](<img width="1280" height="456" alt="IMG-20260521-WA0000" src="https://github.com/user-attachments/assets/571674eb-1e09-4900-b0b2-68d59eddf1ec" />)
---

## Key Findings



---

## Repository Structure

```
carisurg-portfolio/
│
├── README.md                                    ← this file
├── week0_triage_eda.ipynb                       ← main Jupyter notebook
├── EmergencyTriageDataset_Reduced_Dirty.csv     ← raw dataset
│
├── plots/
│   ├── pulse_analytics.png                 ← Day 2 pulse 

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `EmergencyTriageDataset_Reduced_Dirty.csv` to your Google Drive
3. Open `week0_triage_eda.ipynb`
4. Select **Runtime → Restart and run all**
5. The file auto-locates from Drive — no manual path editing needed

**Dependencies** (all pre-installed in Colab):
```
pandas · numpy · matplotlib · seaborn
```

---

## Programme Info

**Programme:** CariSurg MedTech Pathways 2026  

---

*Last updated: Day 2 — Thursday*

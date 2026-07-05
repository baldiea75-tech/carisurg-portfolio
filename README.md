# CariSurg MedTech Pathways - Portfolio

**Programme:** CariSurg MedTech Pathways 2026
**Author:** Ashi Baldie

## Table of Contents
- [About This Project](#about-this-project)
- [Dataset Overview](#dataset-overview)
- [Weekly Progress](#weekly-progress)
  - [Week 0 - Emergency Triage Dataset Cleaning and EDA](#week-0---emergency-triage-dataset-cleaning-and-eda)
  - [Week 4 - Research Proposal: Interpretable Digital Triage System](#week-4---research-proposal-interpretable-digital-triage-system)
  - [Week 5 - Interim Deliverable: Yale EMMLC Data Exploration](#week-5---interim-deliverable-yale-emmlc-data-exploration)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Programme Info](#programme-info)

## About This Project
This repository documents progress through the CariSurg MedTech Pathways Programme, from initial exploratory data analysis (EDA) and clinical data cleaning through to a full research proposal and a second dataset exploration cycle. The overarching goal is to design an interpretable, rule-based digital triage system grounded in recognised clinical frameworks (NEWS2, SIRS, ESI), while surfacing the data quality, equity and workflow risks that any real deployment would need to address.

Week 0 used a dirty simulated triage dataset from Mercer General Hospital's Emergency Department to build clinical data cleaning and visualisation skills. Week 4 translated that groundwork into a full research proposal for a shadow mode pilot at Mercer General. Week 5 opened a second data exploration cycle on a real, published emergency medicine dataset to assess its feasibility for baseline model training.

## Dataset Overview

### Week 0 dataset
**File:** EmergencyTriageDataset_Reduced_Dirty.csv
**Rows:** 2,205 patients
**Columns:** 11

| Column | Raw Dtype | Description | Valid Range |
|--------|-----------|--------------|-------------|
| ID | int | Patient identifier | - |
| Age | int | Patient age (years) | 18-98 in dataset |
| Gender | object | Patient sex | 0 = Female, 1 = Male |
| GCS | object | Glasgow Coma Scale, consciousness | 3-15 |
| SBP | object | Systolic Blood Pressure (mmHg) | 50-250 |
| DBP | float | Diastolic Blood Pressure (mmHg) | 30-150 |
| MAP | float | Mean Arterial Pressure (mmHg) | 40-180 |
| pulse | object | Heart rate (bpm) | 20-250 |
| Temp | object | Body temperature (°C) | 32-43 |
| RR | float | Respiratory Rate (breaths/min) | 5-60 |
| Fio2 | float | Fraction of Inspired Oxygen (%) | 21-100 |

Columns stored as text (object) required type conversion before any numeric analysis was possible.

### Week 5 dataset
**File:** yaleemmlc_admissionprediction_triage.csv
**Rows:** 55,121 encounters
**Columns:** 225 (6 triage vital signs, blood glucose, full demographics, around 200 chief complaint flags, ESI target)

See `data/README.md` for full sourcing details on both datasets.

## Weekly Progress

### Week 0 - Emergency Triage Dataset Cleaning and EDA

#### Day 1 - Gender Column Cleaning
**Date:** Wednesday
**Status:** Complete
**Notebook:** notebooks/week0_triage_eda.ipynb

**What I found**
The Gender column contained 6 variants of two values, caused by inconsistent data entry (mixed case, numeric codes):

| Raw value | Count | Meaning |
|-----------|-------|---------|
| 1 | 422 | Male |
| MALE | 379 | Male |
| Male | 375 | Male |
| FEMALE | 366 | Female |
| Female | 340 | Female |
| 0 | 323 | Female |

**Approach**
Built a mapping dictionary standardising all 6 variants to binary integers (1 = Male, 0 = Female), consistent with the Harvard dataset convention used in this programme.

Used `.map()` instead of `.replace()` so that any unrecognised value would surface as NaN rather than being silently skipped, acting as a validation mechanism. Added an assertion to confirm only {0, 1} exist after cleaning.

**Improvements applied (from feedback)**
- Added inline comments explaining each decision
- Wrapped cleaning logic in a reusable `clean_gender()` function
- Documented assumptions about allowed variants directly in code
- Added post-cleaning assertion to validate the result fails loudly if unexpected values appear

**Result**

| Class | Count |
|-------|-------|
| Male (1) | 1,176 |
| Female (0) | 1,029 |
| Missing (NaN) | 0 |

**Key code**
```python
def clean_gender(series):
    """
    Standardise Gender column to binary integer encoding.
    Convention: 1 = Male, 0 = Female (matches Harvard dataset standard).
    Any unrecognised value becomes NaN surfaced deliberately so issues
    are caught early rather than silently passed through.
    """
    gender_map = {
        'Male': 1, 'MALE': 1, '1': 1,    # all Male variants
        'Female': 0, 'FEMALE': 0, '0': 0  # all Female variants
    }
    return series.map(gender_map)  # unmatched to NaN (intentional)

df['Gender'] = clean_gender(df['Gender'])

# Validate - fails loudly if unexpected categories appear
assert set(df['Gender'].dropna().unique()) == {0, 1}, \
    f"Unexpected Gender values found: {df['Gender'].unique()}"
assert df['Gender'].isnull().sum() == 0, "Unexpected NaNs in Gender after cleaning"
```

#### Day 2 - Pulse Column Cleaning
**Date:** Thursday
**Status:** Complete
**Notebook:** notebooks/week0_triage_eda_day2.ipynb

**What I found**
The pulse (heart rate) column was stored as object (string) dtype despite being a numeric measurement. After `pd.to_numeric()` conversion:
- 44 NaNs created from non-numeric text entries
- 43 additional rows had physiologically impossible values: 0.0 (impossible for a living patient) and 300.0 (far above any survivable heart rate)
- Valid range used: 20-250 bpm (per CariSurg clinical reference table)

**Approach**
- Converted to numeric with `errors='coerce'`, non-numeric text becomes NaN
- Flagged and replaced physiologically impossible values (below 20 or above 250) with NaN
- Imputed all NaNs using the median (90.0 bpm), chosen over mean because heart rate data is right-skewed with outliers
- Companion-cleaned SBP in the same step to enable the composite shock index flag
- Wrapped each stage in a reusable function with inline clinical justification

**Clinical significance**
Heart rate is critical for detecting:
- Tachycardia (above 100 bpm): fever, pain, blood loss or anxiety
- Shock index: when combined with SBP below 90, indicates haemodynamic compromise
- Bradycardia (below 60 bpm): heart block or medication effect

**Composite Shock Flag**

| SBP | Pulse | Interpretation |
|-----|-------|----------------|
| below 90 mmHg | above 110 bpm | Suspected decompensated shock |

Result: 32 patients (1.45%) flagged as potential shock cases.

**Result after cleaning**

| Metric | Value |
|--------|-------|
| Min | 40.0 bpm |
| Median | 90.0 bpm |
| Max | 170.0 bpm |
| NaNs remaining | 0 |

**Visualisation:** plots/pulse_analytics.png

#### Day 3 - Data Visualisation
**Date:** Friday
**Status:** Complete
**Notebook:** notebooks/week0_triage_eda_day3.ipynb

**Overview**
Produced six clinically meaningful plots from the fully cleaned triage dataset. Every plot answers a specific clinical question. Cleaning from Days 1 and 2 was consolidated into a single reusable `apply_all_cleaning()` function and called once before any plotting. A seventh custom plot was added as the open ended exercise.

**Plots produced**

| File | Plot type | Clinical question answered |
|------|-----------|------------------------------|
| gender_distribution.png | Bar chart | What is the sex breakdown of triage patients? |
| gcs_histogram.png | Histogram | Are most patients fully alert or many presenting with reduced consciousness? |
| pulse_histogram.png | Histogram | What proportion present with abnormal heart rates? |
| temp_histogram.png | Histogram | What proportion present with fever or hypothermia? |
| rr_histogram.png | Histogram | What proportion present with abnormal breathing rates? |
| sbp_vs_dbp.png | Scatter | Does the SBP/DBP relationship follow expected physiology? |
| age_vs_pulse.png | Scatter | Does heart rate vary with patient age? |
| vitals_boxplots.png | Box plots (multi panel) | Do all vital sign distributions sit within plausible ranges post cleaning? |
| my_plot.png | Scatter (custom) | Do fever and tachypnoea co-occur and does the pattern suggest sepsis or pneumonia? |

**Custom plot - Temperature vs Respiratory Rate**
Clinical question: Do fever and tachypnoea co-occur, and is this pattern more consistent with sepsis or pneumonia?

Both fever (Temp above 38.0°C) and tachypnoea (RR above 20/min) are SIRS criteria, Systemic Inflammatory Response Syndrome. When a patient meets 3 or more SIRS criteria the working clinical diagnosis shifts toward sepsis rather than a localised infection like pneumonia.

| Pattern | Criteria | Patients flagged |
|---------|----------|-------------------|
| Dual SIRS flag | Fever and tachypnoea | 133 (6.0%) |
| SIRS 3 or more criteria | Fever and tachypnoea and HR above 100 | 88 (4.0%) |
| Pneumonia like | Fever and tachypnoea and FiO2 above 21% | 29 (1.3%) |

The data supports sepsis rather than pneumonia. 88 of the 133 dual flag patients also had tachycardia (HR above 100 bpm), meeting three SIRS criteria. Only 29 patients showed the pneumonia pattern, a much smaller subset.

**Temperature histogram findings**
- Hypothermia (below 36.0°C): 22 patients (1.0%)
- Low grade fever (above 37.5°C): 475 patients (21.5%)
- Fever, SIRS criterion (above 38.0°C): 282 patients (12.8%)
- High fever (above 39.0°C): 86 patients (3.9%)

**Respiratory Rate histogram findings**
- Bradypnoea (below 12/min): 0 patients (0.0%)
- Tachypnoea (above 20/min): 576 patients (26.1%)
- Severe tachypnoea (above 30/min): 131 patients (5.9%)

Over 1 in 4 patients arrived with an elevated respiratory rate, consistent with the high SIRS burden identified in the custom scatter plot.

#### Day 4 - Clinical Write-Up: Respiratory Rate
**Date:** 23 May 2026
**Status:** Complete
**Document:** docs/RR Clincal Write Up (1).pdf

**Summary**
Respiratory Rate (RR) is the number of breaths a person takes per minute, measured by counting chest wall movements over 60 seconds. It reflects the body's effort to maintain adequate gas exchange and acid base balance, making it sensitive to disruptions across multiple organ systems rather than the lungs alone (Porter and Graham, 2025). The normal range for a resting adult is 12 to 20 breaths per minute.

A rate below 12 breaths per minute is termed bradypnea and is associated with opioid or sedative use, traumatic brain injury or alcohol intoxication. A rate above 20 breaths per minute is termed tachypnea and may indicate pneumonia, sepsis, pulmonary embolism, metabolic acidosis or cardiac dysfunction (Porter and Graham, 2025).

**Clinical importance at triage**
RR holds particular clinical weight because it is frequently the earliest objective indicator of patient deterioration, rising before blood pressure or oxygen saturation falls (Cretikos et al., 2008). It is one of the three qSOFA criteria, where a rate of 22 breaths per minute or more contributes to identifying patients at high risk of sepsis related death or prolonged intensive care admission (Perman et al., 2020).

Despite its importance, RR remains the vital sign most prone to measurement error in emergency settings, with studies showing poor agreement between manual and automated recording (Lee et al., 2024).

**Connection to dataset**
The Mercer General dataset showed 576 patients (26.1%) with tachypnoea on arrival and 131 (5.9%) in the severe range above 30 breaths per minute. The RR histogram produced in Day 3 visualises this burden directly. RR also featured in both the SIRS flag logic and the custom scatter plot, where it was paired with temperature to identify patients at risk of sepsis.

**References**
- Cretikos, M. A., Bellomo, R., Hillman, K., Chen, J., Finfer, S., and Flabouris, A. (2008). Respiratory rate: the neglected vital sign. *The Medical Journal of Australia*, 188(11), 657-659.
- Lee, J. H., Nathanson, L. A., Burke, R. C., Anthony, B. W., Shapiro, N. I., and Dagan, A. S. (2024). Assessment of respiratory rate monitoring in the emergency department. *Journal of the American College of Emergency Physicians Open*, 5(3), e13154.
- Perman, S. M., Mikkelsen, M. E., Goyal, M., Ginde, A., Bhardwaj, A., Drumheller, B., Sante, S. C., Agarwal, A. K., and Gaieski, D. F. (2020). The sensitivity of qSOFA calculated at triage and during emergency department treatment to rapidly identify sepsis patients. *Scientific Reports*, 10(1), 20395.
- Porter, R., and Graham, D. D. (2025). Abnormal Respirations. In *StatPearls*. StatPearls Publishing.

#### Day 5 - Unconsidered Metrics: SpO2
**Date:** 24 May 2026
**Status:** Complete
**Document:** docs/SpO2 - Unconsidered Metrics.pdf

**Summary**
Peripheral oxygen saturation (SpO2) is a non invasive measure of how much haemoglobin in the blood is carrying oxygen, expressed as a percentage and recorded using a pulse oximeter placed on the fingertip or earlobe (Cleveland Clinic, 2022). In a healthy adult a normal SpO2 reading falls between 95% and 98%, while readings below 94% are clinically significant and anything below 90% constitutes a medical emergency (Piraino et al., 2022).

**Why SpO2 is missing and why it matters**
The Mercer General triage dataset includes FiO2 (the fraction of inspired oxygen being delivered) but does not include SpO2 (whether that oxygen is actually reaching the blood effectively). This is a meaningful gap because FiO2 alone cannot confirm adequate oxygenation. A patient on 100% supplemental oxygen may still be critically hypoxic if their lungs are too damaged to transfer it.

Studies have shown that even mild drops in blood oxygen levels, from 90% to 93%, are associated with significantly higher mortality than readings above 95% (Graham et al., 2024). In triage, low SpO2 has been identified as one of the vital signs most strongly associated with mortality, alongside abnormal blood pressure and respiratory rate (Simbawa et al., 2021).

**Impact on the triage model**
Because the dataset lacks SpO2, any model built from it can only track the oxygen treatment being delivered rather than whether the patient's body is absorbing it. This was cited as one of the key limitations in the Day 6 pseudocode: a production ready version of the digital triage system would need to collect SpO2 at the point of triage to enable accurate hypoxia detection.

**References**
- Cleveland Clinic. (2022, February 18). Blood oxygen level: What it is and how to increase it.
- Graham, H. R., King, C., Duke, T., Ahmed, S., Baqui, A. H., Colbourn, T., Falade, A. G., Hildenwall, H., Hooli, S., Kamuntu, Y., Subhi, R., and McCollum, E. D. (2024). Hypoxaemia and risk of death among children: rethinking oxygen saturation, risk stratification, and the role of pulse oximetry in primary care. *The Lancet Global Health*, 12(8), e1359-e1364.
- Piraino, T., Madden, M., Roberts, K. J., Lamberti, J., Ginier, E., and Strickland, S. L. (2022). AARC clinical practice guideline: Management of adult patients with oxygen in the acute care setting. *Respiratory Care*, 67(1), 115-128.
- Simbawa, J. H., Jawhari, A. A., Almutairi, F., Almahmoudi, A., Alshammrani, B., Qashqari, R., and Alattas, I. (2021). The association between abnormal vital signs and mortality in the emergency department. *Cureus*, 13(12), e20454.

#### Day 6 - Digital Triage Pseudocode
**Date:** 24 May 2026
**Status:** Complete
**Document:** docs/Triage Pseudocode.pdf

**Overview**
Designed a rule based digital triage system grounded in the NEWS2 (National Early Warning Score 2) framework, which was mandated across NHS England as the standard early warning system for identifying acutely ill patients including those with sepsis (Royal College of Physicians, 2017). Machine learning research has shown strong predictive performance for triage models (Tschoellitsch et al., 2023), but a rule based approach was chosen for this first version because it is more explainable and safer to deploy in a clinical setting.

**System steps**
The system processes one patient row at a time through seven sequential steps:
1. Receive raw data from the electronic health record or nurse tablet entry
2. Validate and clean each input using the same range filters and median imputation established in Days 1 and 2, raising a `missing_flag` if any critical vital is imputed to warn the clinician that output confidence is reduced (Miles et al., 2020)
3. Calculate NEWS2 score by allocating 0 to 3 points per vital sign based on deviation from normal, plus 2 points for supplemental oxygen (RCP, 2017)
4. Run SIRS sepsis flag as a secondary safety layer alongside NEWS2, checking for fever, tachycardia and tachypnoea, since this combination improves sepsis sensitivity at triage (Dewitte et al., 2022)
5. Check composite shock flag using the pattern identified in Day 2: SBP below 90 mmHg and heart rate above 110 bpm
6. Assign risk level combining all three outputs, where any positive safety flag escalates the risk category upward regardless of the NEWS2 score alone
7. Output to triage dashboard with Patient ID, NEWS2 score, SIRS criteria count, shock flag, final risk level, reason and a data completeness indicator

**Risk categories**

| Risk Level | NEWS2 Score | Clinical Response |
|------------|-------------|---------------------|
| Low | 0 to 4 | Routine monitoring, reassess as scheduled |
| Low Medium | Red flag* | Inform nurse in charge, increase monitoring |
| Medium | 5 to 6 | Urgent review by doctor or acute care team |
| High | 7 or above | Immediate review, consider ICU escalation |

*Red flag = any single parameter scoring 3 points (GCS below 13, SBP below 90, RR of 25 or above) regardless of total score. If shock or SIRS flag is also present at High risk, the sepsis bundle is automatically activated.

**Limitations**
This system is decision support only and cannot replace a nurse's direct assessment (Tschoellitsch et al., 2023). The dataset also lacks SpO2, white blood cell count and chief complaint, all of which would be required for a production ready version (as identified in Day 5).

**References**
- Dewitte, K., Scheurwegs, E., Van Ierssel, S., Jansens, H., Dams, K., and Roelant, E. (2022). Audit of a computerized version of the Manchester triage system and a SIRS based system for the detection of sepsis at triage in the emergency department. *International Journal of Emergency Medicine*, 15(1), 67.
- Miles, J., Turner, J., Jacques, R., Williams, J., and Mason, S. (2020). Using machine learning risk prediction models to triage the acuity of undifferentiated patients entering the emergency care system: a systematic review. *Diagnostic and Prognostic Research*, 4, 16.
- Royal College of Physicians. (2017). National Early Warning Score (NEWS) 2: Standardising the assessment of acute illness severity in the NHS. RCP.
- Smith, G. B., Redfern, O. C., Pimentel, M. A., Gerry, S., Collins, G. S., Malycha, J., Prytherch, D., Schmidt, P. E., and Watkinson, P. J. (2019). The National Early Warning Score 2 (NEWS2). *Clinical Medicine*, 19(3), 260.
- Tschoellitsch, T., Seidl, P., Bock, C., Maletzky, A., Moser, P., Thumfart, S., Giretzlehner, M., Hochreiter, S., and Meier, J. (2023). Using emergency department triage for machine learning based admission and mortality prediction. *European Journal of Emergency Medicine*, 30(6), 408-416.

### Week 4 - Research Proposal: Interpretable Digital Triage System
**Date:** 13 June 2026
**Status:** Complete
**Document:** docs/Week 4 Final Deliverable.pdf
**Title:** Development of an Interpretable Digital Triage System for Dynamic Risk Assessment in Emergency Care

**Overview**
Week 4 translated the Week 0 groundwork into a full 12 week research proposal for a shadow mode pilot of an interpretable digital triage system at Mercer General Hospital. The proposal responds to two gaps identified across the literature review: the near universal reliance on static, single point triage assessment, and the persistent failure of high performing but uninterpretable AI triage systems to reach clinical deployment.

**Statement of the problem**
Emergency department overcrowding delays care for critically ill patients because traditional triage relies on static, point in time assessments that fail to capture real time physiological deterioration in the waiting room, while current AI triage solutions operate as uninterpretable black boxes that hinder clinical adoption.

**Proposed solution: four components**

| Component | Purpose |
|-----------|---------|
| Multi Source Ingestion Engine | Combines structured triage desk vitals with an NLP scan of free text nurse notes to flag high risk clinical language |
| Time Delta Trajectory Tracking | Issues a re-triage prompt after 30 minutes without a vitals update and flags directional change (trajectory) rather than a single threshold reading |
| Glass Box Logic Layer | Maps every risk output onto NEWS2 and SIRS language clinicians already recognise instead of a raw probability score |
| Parallel Shadow Workflow Pilot | Runs the system silently alongside existing manual triage for 12 weeks, logging divergence and time advantage without influencing care decisions |

**Methodology**
The workflow diagram was adapted from the only qualitative observational study of patient flow conducted in a Caribbean ED to date (De Freitas et al., 2020), which mapped 143 patient journeys across 203 hours of direct observation in Trinidad and Tobago. The diagram traces the patient journey from arrival to ward transfer, overlaying five AI plug in points and three workflow constraints (physical separation of registration and triage, nursing staffing shortages, and competing demands on porter and transfer staff) onto that validated structure.

**Risk register**
The proposal includes a full risk register covering nine named risks across equity, technical, ethical and operational categories, including training distribution mismatch between Mercer General and underserved Caribbean populations, alert fatigue from high frequency re-triage prompts, automation bias toward NEWS2 output, and NLP misclassification of Creole influenced language. Each risk carries a likelihood and impact rating, a mitigation and a measurable signal of success.

**Real world AI harm case study**
The proposal examines the 2019 Optum racial bias algorithm (Obermeyer et al., 2019), in which a US healthcare algorithm used predicted healthcare spending as a proxy for health need and systematically assigned Black patients lower risk scores than equally sick White patients. The case study draws three design lessons directly into the Mercer General pilot: the shadow mode design removes the immediate harm pathway, the risk register requires a stratified equity audit at weeks 4, 8 and 12 before any deployment recommendation, and the Glass Box Logic Layer ensures every output is traceable to a specific vital sign or nurse note.

**Literature base**
The proposal is grounded in 25 peer reviewed references spanning AI triage performance, explainable AI and clinician trust, dynamic early warning scores, Caribbean ED workflow research and documented algorithmic bias in healthcare AI.

### Week 5 - Interim Deliverable: Yale EMMLC Data Exploration
**Date:** 4 July 2026
**Status:** In progress (interim)
**Notebook:** notebooks/Week_5_Interim_deliverable.ipynb
**Dataset:** Yale EMMLC Admission Prediction Triage (yaleemmlc_admissionprediction_triage.csv)

**Overview**
Week 5 opened a second data exploration cycle on a real, published emergency medicine dataset (55,121 encounters, 225 columns) to assess its feasibility for baseline model training ahead of Week 6. The target variable is the Emergency Severity Index (ESI), a nurse assigned acuity level from 1 (most urgent) to 5 (least urgent).

**Column families defined**

| Family | Description |
|--------|--------------|
| Vitals | 7 columns: heart rate, systolic and diastolic BP, respiratory rate, oxygen saturation, temperature (Fahrenheit), blood glucose |
| Demographics | 9 columns: age, gender, ethnicity, race, language, religion, marital status, employment status, insurance status |
| Admin | 5 columns: department name, arrival mode, arrival month, arrival day, arrival hour bin |
| Leakage | 2 columns: disposition and previousdispo, known only after the visit ends, excluded from all feature sets |
| Chief complaint | around 200 binary flag columns, one per presenting complaint |

**ESI target distribution**
ESI 2 and 3 together account for 81.5% of encounters. ESI 1, the most critical level, represents only 0.1% (77 patients out of 55,121), an extreme class imbalance that will need stratified sampling or class weighting in Week 6.

**Data quality findings**
- Zero missing values across all 225 columns, unusually clean for a clinical dataset, though clean data from a single site can still carry systematic bias
- Temperature is recorded in Fahrenheit rather than Celsius (mean 98.09F), so fever thresholds must use 100.4F rather than 38C
- 4 respiratory rate values above 60/min and 25 blood glucose values above 800 mg/dL flagged as outliers, to be replaced with NaN and imputed with the column median
- 174 of 200 chief complaint flags appear in fewer than 1% of patients and are candidates for removal as near zero variance columns before modelling

**Correlation with ESI**
Age is the strongest correlate (r = -0.237), consistent with older patients tending to be more acutely unwell. Oxygen saturation shows the expected positive relationship (r = +0.178), where a higher reading indicates lower urgency.

**Plots produced**

| File | Plot type | Clinical question answered |
|------|-----------|------------------------------|
| plot1_esi_distribution.png | Bar chart | How are triage acuity levels distributed across 55,121 encounters? |
| plot2_demographics.png | Bar charts (race and insurance) | How representative is this dataset for a Caribbean ED? |
| plot3_vitals_distributions.png | Histograms (6 panel) | Do cleaned vital sign distributions look physiologically plausible? |
| plot4_chief_complaints.png | Horizontal bar chart | Which presenting complaints drive triage volume? |

**Feasibility memo outline**
A feasibility memo was drafted for the ED Board covering the verdict, dataset summary, top three concerns (leakage columns, ESI 1 class imbalance, US academic hospital data applied to a Caribbean ED), top three reasons to proceed, caveats and a ranked shortlist of ten candidate features for Week 6 modelling, led by age, oxygen saturation and respiratory rate.

## Repository Structure
```
carisurg-portfolio/
|
|-- README.md
|-- data/
|   |-- README.md
|-- notebooks/
|   |-- README.md
|   |-- week0_triage_eda.ipynb
|   |-- week0_triage_eda_day2.ipynb
|   |-- week0_triage_eda_day3.ipynb
|   |-- Week_1_Final_Deliverable.ipynb
|   |-- Week_5_Interim_deliverable.ipynb
|-- docs/
|   |-- README.md
|   |-- A. Baldie Career Deck.pptx
|   |-- RR Clincal Write Up (1).pdf
|   |-- SpO2 - Unconsidered Metrics.pdf
|   |-- Triage Pseudocode.pdf
|   |-- Week 1_ Interim Deliverable.pdf
|   |-- Week 1_ Final Deliverable.pdf
|   |-- Week 2 Final Proposal.pdf
|   |-- Week3 Workflow Mapping.pdf
|   |-- Week 4 Final Deliverable.pdf
|-- plots/
    |-- gender_distribution.png
    |-- gcs_histogram.png
    |-- pulse_histogram.png
    |-- temp_histogram.png
    |-- rr_histogram.png
    |-- sbp_vs_dbp.png
    |-- age_vs_pulse.png
    |-- vitals_boxplots.png
    |-- my_plot.png
    |-- pulse_analytics.png
    |-- triage_workflow_diagram.png
    |-- plot1_esi_distribution.png
    |-- plot2_demographics.png
    |-- plot3_vitals_distributions.png
    |-- plot4_chief_complaints.png
```

## How to Run
1. Open Google Colab
2. Upload the relevant dataset (EmergencyTriageDataset_Reduced_Dirty.csv or yaleemmlc_admissionprediction_triage.csv) to your Google Drive
3. Open the relevant notebook from the `notebooks/` folder
4. Select Runtime > Restart and run all
5. The file auto locates from Drive, no manual path editing needed

**Dependencies** (all pre installed in Colab): pandas, numpy, matplotlib, seaborn

## Programme Info
**Programme:** CariSurg MedTech Pathways 2026

**Last updated:** Saturday 4 July 2026

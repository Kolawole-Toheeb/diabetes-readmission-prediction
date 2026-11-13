# Diabetes Hospital Readmission Prediction

## Problem description

When a patient with diabetes leaves hospital, some are admitted again within 30 days. Early readmission is costly for hospitals and bad for patients. If a hospital knew which patients are most at risk when they are discharged, it could give those patients extra follow-up, such as a phone call, a clinic visit or a medication review.

This project builds a machine learning model that estimates the **probability that a diabetic patient is readmitted within 30 days of discharge**. It is served as a web service: a hospital system sends a patient's details and gets back a probability. The model is a screening aid that flags higher-risk patients. It does not replace a doctor's decision.

## Data

The data is the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/296). It has about 101,766 hospital visits and 50 columns. A copy is included in this repo at `data/diabetic_data.csv`.

Target: `readmit_30` = 1 if the patient was readmitted in less than 30 days, otherwise 0. About 9% of patients are in the positive class.

## Files

| File | What it does |
|---|---|
| `notebook.ipynb` | Data cleaning, EDA, feature importance, model comparison and tuning |
| `train.py` | Prepares the data, trains the final model and saves it to `model.bin` |
| `predict.py` | FastAPI web service that loads `model.bin` and returns a probability |
| `Dockerfile` | Runs the web service in a container |
| `pyproject.toml`, `uv.lock` | Dependencies (managed with `uv`) |
| `data/diabetic_data.csv` | The dataset |

## What I did

1. **Cleaning:** removed patients who died or went to hospice (they cannot be readmitted), kept one visit per patient (so the same patient is never in both train and test), dropped `weight` (97% missing), and filled other missing values with "Unknown". Diagnosis codes were grouped into broad disease groups.
2. **EDA:** readmission is more common in older patients, in patients with more earlier inpatient visits and in patients with longer hospital stays. The most informative features were where the patient went after discharge (`discharge_disposition_id`) and the number of earlier inpatient visits (`number_inpatient`).
3. **Models:** I compared logistic regression, decision tree, random forest and XGBoost, and tuned each one on a validation set.
4. **Result:** XGBoost (`eta=0.1`, `max_depth=5`) was best, with a validation AUC of 0.642 and a **test AUC of 0.653**. This is a modest score, which is normal for readmission prediction, because many things that affect readmission are not in the data.

| Model | Best validation AUC |
|---|---|
| XGBoost | 0.642 |
| Random forest | 0.639 |
| Logistic regression | 0.636 |
| Decision tree | 0.615 |

## How to run

### 1. Install the dependencies

The project uses [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
uv sync
```

### 2. Train the model

```bash
uv run python train.py
```

This prints the test AUC and saves the model to `model.bin`.

### 3. Run the web service locally

```bash
uv run uvicorn predict:app --port 9696
```

### 4. Run it with Docker

```bash
docker build -t diabetes-readmission .
docker run -d -p 9696:9696 --name dr diabetes-readmission
```

### 5. Send a request

```bash
curl -s localhost:9696/predict -H 'Content-Type: application/json' -d '{"race":"Caucasian","gender":"Female","age":"[70-80)","admission_type_id":"1","discharge_disposition_id":"1","admission_source_id":"7","diag_1":"Circulatory","diag_2":"Diabetes","diag_3":"Other","time_in_hospital":5,"number_inpatient":3}'
```

Expected reply: `{"readmission_probability":0.0753}`

To stop the container: `docker stop dr`

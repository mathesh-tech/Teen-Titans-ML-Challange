# Business Entity Resolution — Amazon ML Challenge 2026

A clean, scalable, OOP-based Python project skeleton for the Business Entity
Resolution task. This skeleton establishes the project structure, logging,
configuration, and data loading/validation. Advanced logic (cleaning,
candidate generation, feature engineering, model training) is stubbed out
and will be implemented in later iterations.

## Project Structure

```
project/
├── dataset/                   # Place raw train.tsv / test.tsv here
├── output/                    # Predictions / submission files written here
├── models/                    # Trained model artifacts saved here
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # TSV loading + basic validation (IMPLEMENTED)
│   ├── data_cleaner.py        # Text/address normalization (STUB)
│   ├── candidate_generator.py # Blocking + candidate pair generation (STUB)
│   ├── feature_engineer.py    # Similarity feature computation (STUB)
│   ├── model_trainer.py       # LightGBM training/eval/persistence (STUB)
│   ├── predictor.py           # Inference + submission generation (STUB)
│   └── utils.py                # Logging setup + ProjectConfig (IMPLEMENTED)
├── train.py                   # Training pipeline entry point
├── predict.py                 # Inference pipeline entry point
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.11+**
- **Pandas** — data loading and manipulation
- **Scikit-learn** — model evaluation utilities, splitting, metrics
- **RapidFuzz** — fast fuzzy string matching for candidate generation & features
- **LightGBM** — gradient boosted model for match/no-match classification

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3.11 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data
#    Put train.tsv and test.tsv inside the dataset/ folder
```

## Workflow

The intended end-to-end pipeline (target design):

1. **Data Loading** (`DataLoader`) — ✅ Implemented
   Load `train.tsv` / `test.tsv`, validate structure (non-empty, required
   columns present), and log dataset shape/stats.

2. **Data Cleaning** (`DataCleaner`) — 🚧 Stub
   Normalize business names/addresses, handle missing values, remove
   duplicates, and standardize text fields before comparison.

3. **Candidate Generation** (`CandidateGenerator`) — 🚧 Stub
   Reduce the full pairwise comparison space using blocking keys and
   RapidFuzz-based approximate matching, producing a manageable set of
   candidate record pairs.

4. **Feature Engineering** (`FeatureEngineer`) — 🚧 Stub
   Compute similarity features for each candidate pair: string similarity
   (Levenshtein, Jaro-Winkler, token-based scores via RapidFuzz), token
   overlap, and numeric/categorical field comparisons.

5. **Model Training** (`ModelTrainer`) — 🚧 Stub
   Train a LightGBM binary classifier (match / no-match) on the engineered
   features, evaluate via cross-validation, and persist the trained model
   to `models/`.

6. **Prediction** (`Predictor`) — 🚧 Stub
   Load the trained model, score candidate pairs from the test set, and
   write a submission-ready file to `output/`.

## Running the Pipeline

```bash
# Run the training pipeline
python train.py

# Run the inference pipeline
python predict.py
```

Currently, both scripts will:
- Load configuration and set up logging
- Load and validate the TSV dataset from `dataset/`
- Instantiate all pipeline stage objects (cleaner, candidate generator,
  feature engineer, trainer/predictor)
- Log that downstream stages are not yet implemented, then exit cleanly

## Configuration

All paths and pipeline-wide settings are centralized in
`src/utils.py::ProjectConfig`. Instantiate it via `load_config()`, and
override any field as needed:

```python
from src.utils import load_config

config = load_config(train_file="my_train.tsv", log_level=10)  # DEBUG level
```

## Next Steps (Not Yet Implemented)

- [ ] Implement text normalization and address standardization in `DataCleaner`
- [ ] Implement blocking + RapidFuzz candidate retrieval in `CandidateGenerator`
- [ ] Implement similarity/overlap/numeric features in `FeatureEngineer`
- [ ] Implement LightGBM training, cross-validation, and persistence in `ModelTrainer`
- [ ] Implement inference and submission formatting in `Predictor`
- [ ] Add unit tests for each module

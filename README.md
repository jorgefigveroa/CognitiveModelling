# Cognitive Modelling — Encoding of Facial Features

A full pipeline for a cognitive modelling group project at DTU. Participants rate greyscale face images on a 1–5 scale. Their ratings are used to build a PCA-based linear encoding model that predicts perceived facial features and generates synthetic face images along a rating continuum.

---

## Project Structure

```
CognitiveModelling/
├── images/
│   └── greyscale/                  # Preprocessed greyscale face images
├── results/
│   ├── Ania.csv                    # Experiment 1 ratings
│   ├── Jorge.csv
│   ├── maria.csv
│   ├── pca/
│   │   ├── average_face.png        # Mean face across all images
│   │   ├── variance_explained.png  # Bar plot of variance per PC
│   │   ├── cumulative_variance.png # Cumulative variance curve
│   │   ├── PC1.png – PC5.png       # First 5 PC visualisations (min/avg/max)
│   │   ├── pca_scores_48.csv       # PC scores (48 PCs) for all images
│   │   └── pca_model.npz           # Saved mean image + components
│   ├── regression/
│   │   ├── ratings_*.png           # Rating histograms per participant
│   │   ├── forward_selection.csv   # CV MSE at each forward selection step
│   │   ├── selected_pc_coefficients.csv  # Final model coefficients
│   │   ├── model_data.csv          # Merged ratings + PC scores
│   │   └── linear_model.npz        # Saved regression model
│   ├── synthetic_faces/            # Original synthetic faces (0.5–5.5 range)
│   ├── synthetic_faces_6/          # Point 6: within-distribution synthetic faces
│   │   ├── faces/                  # 11 PNG faces (rating 1.22–4.22)
│   │   ├── face_target_ratings.csv # Filename → target rating map
│   │   ├── contact_sheet_within_range.png
│   │   ├── predicted_rating_distribution.png
│   │   └── summary.csv
│   ├── synthetic_faces_7/          # Point 7: analysis of Experiment 2
│   │   ├── boxplots.png            # Box plots per participant
│   │   ├── mean_rating_vs_predicted.png
│   │   └── spearman_results.csv
│   └── experiment_2/               # Experiment 2 ratings (one CSV per participant)
├── greyscale.py                    # Downloads and converts images to greyscale
├── experiment.py                   # Experiment 1: rate original face images
├── experiment_2.py                 # Experiment 2: rate synthetic face images
├── pca.py                          # PCA on face images, PC visualisation
├── regression.py                   # Forward selection + linear regression
├── synthetic_faces.py              # Generate synthetic faces (step 5)
├── synthetic_faces_6.py            # Distribution check + within-range faces (step 6)
├── synthetic_faces_7.py            # Analyse Experiment 2 results (step 7)
├── pyproject.toml
└── README.md
```

---

## Setup

Requires Python 3.13+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install pillow gdown matplotlib numpy pandas scikit-learn scipy
```

---

## Pipeline

Run the scripts in this order. All scripts must be run from the project root (`CognitiveModelling/`).

### 1. Preprocess images

Only needed if you want to regenerate images from Google Drive:

```bash
uv run python greyscale.py
```

Downloads face images (ages 24–27 from UTKFace), converts to greyscale, saves to `images/greyscale/`.

---

### 2. Run Experiment 1

Each participant runs the script once:

```bash
uv run python experiment.py
```

- Enter the participant name when prompted.
- Press **1–5** to rate each face (1 = low, 5 = high).
- Each image is shown **twice** in randomised order.
- Results saved to `results/<name>.csv` with columns `filename`, `rating_1`, `rating_2`.

---

### 3. PCA and dimension reduction

```bash
uv run python pca.py
```

- Loads all greyscale images and subtracts the mean image (no standardisation by std).
- Runs PCA on the centred image matrix.
- Saves the first 5 PC visualisations as triplets: `mean + min_score * PC | mean | mean + max_score * PC`.
- Saves a bar plot of variance explained and a cumulative variance curve.
- Saves the first **48 PC scores** per image to `results/pca/pca_scores_48.csv`.
- Saves the PCA model (mean image + components) to `results/pca/pca_model.npz`.

---

### 4. Linear regression with forward selection

```bash
uv run python regression.py
```

- Loads PCA scores and all participant ratings from `results/`.
- Converts ratings to long format and merges with PC scores.
- Runs **forward selection** with 5-fold `GroupKFold` cross-validation (grouped by image filename to prevent leakage).
- Selects 24 PCs out of 48 candidates.
- Fits the final `LinearRegression` model on the selected PCs.
- Saves the model to `results/regression/linear_model.npz`.

---

### 5. Generate synthetic faces

```bash
uv run python synthetic_faces.py
```

- Generates 11 synthetic faces at ratings 0.5, 1.0, 1.5, …, 5.5 using the formula:
  `α = (target_rating − intercept) / ‖w‖²`, `j̃₀ = α · w`
- Saves faces and a contact sheet to `results/synthetic_faces/`.

---

### 6. Distribution check and within-range faces

```bash
uv run python synthetic_faces_6.py
```

- Predicts ratings for every training image using the fitted model.
- Computes the **observed rating range**: min = **1.22**, max = **4.22**.
- Compares against the requested range (0.5–5.5): both ends fall outside the distribution.
- Generates **11 new synthetic faces** evenly spaced within the observed range (1.22–4.22).
- Saves a contact sheet, histogram, and a `face_target_ratings.csv` map for Experiment 2.

---

### 7. Run Experiment 2

Each participant runs the script once:

```bash
uv run python experiment_2.py
```

- Enter the participant name when prompted.
- Press **1–5** to rate each synthetic face.
- Each of the 11 faces is shown **10 times** in randomised order (110 trials total).
- Results saved to `results/experiment_2/<name>.csv` with columns `filename`, `trial`, `rating`.

---

### 8. Analyse Experiment 2 results

```bash
uv run python synthetic_faces_7.py
```

- Loads all CSVs from `results/experiment_2/`.
- Joins each rating to its model-predicted target rating via `face_target_ratings.csv`.
- Produces:
  - `boxplots.png` — one subplot per participant showing rating distribution per predicted rating, with Spearman ρ in each title.
  - `mean_rating_vs_predicted.png` — mean ± SEM across participants vs the identity line.
  - `spearman_results.csv` — Spearman ρ and p-value per participant.

---

## Output files summary

| File | Generated by | Description |
|---|---|---|
| `results/pca/pca_scores_48.csv` | `pca.py` | 48 PC scores per image |
| `results/pca/pca_model.npz` | `pca.py` | Mean image + PC components |
| `results/regression/linear_model.npz` | `regression.py` | Intercept + coefficients for selected PCs |
| `results/synthetic_faces_6/faces/` | `synthetic_faces_6.py` | 11 within-range synthetic face PNGs |
| `results/synthetic_faces_6/face_target_ratings.csv` | `synthetic_faces_6.py` | Filename → target rating map |
| `results/experiment_2/<name>.csv` | `experiment_2.py` | Experiment 2 ratings per participant |
| `results/synthetic_faces_7/boxplots.png` | `synthetic_faces_7.py` | Box plots + Spearman ρ |
| `results/synthetic_faces_7/spearman_results.csv` | `synthetic_faces_7.py` | Spearman ρ and p-values |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pillow` | Image loading, greyscale conversion, contact sheets |
| `gdown` | Downloading from Google Drive |
| `numpy` | Numerical computation, PCA, image reconstruction |
| `pandas` | Data loading and manipulation |
| `matplotlib` | All plots and visualisations |
| `scikit-learn` | PCA, LinearRegression, GroupKFold |
| `scipy` | Spearman rank correlation |
| `tkinter` | GUI for experiments (included with Python) |

---

## Image Dataset

Face images from the **UTKFace** dataset, ages 24–27.

Filename format: `<age>_<gender>_<race>_<timestamp>.jpg.chip.jpg`
- Gender: `0` = male, `1` = female
- Race: `0` = White, `1` = Black, `2` = Asian, `3` = Indian, `4` = Other

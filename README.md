# Cognitive Modelling — Approachability & Attractiveness Experiment

A behavioural experiment that measures perceived approachability and attractiveness of face images across different age groups. Participants rate greyscale face images on a 1–5 scale using a keyboard-driven GUI. Each image is presented twice per session and results are saved per participant as CSV.

---

## Project Structure

```
CognitiveModelling/
├── images/
│   └── greyscale/          # Preprocessed greyscale face images (committed)
├── results/                # Per-participant CSV output (generated at runtime)
├── greyscale.py            # Downloads images from Google Drive and converts to greyscale
├── experiment.py           # Runs the rating experiment (tkinter GUI)
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
pip install pillow gdown
```

---

## Usage

### 1. Preprocess images (already done — greyscale images are in the repo)

Only needed if you want to regenerate the images from Google Drive:

```bash
uv run python greyscale.py
```

This downloads face images from Google Drive (ages 24–27), converts them to greyscale at original resolution, and saves them to `images/greyscale/`. The originals are never stored — a temporary directory is used and auto-deleted after processing.

### 2. Run the experiment

```bash
uv run python experiment.py
```

- Enter the participant's name when prompted.
- A window will open displaying one face image at a time.
- **Press 1–5** to rate the image (1 = not approachable/attractive, 5 = very approachable/attractive).
- Each image is shown **twice** in a randomised order.
- When all trials are complete, results are saved automatically.

---

## Output

Results are saved to `results/<participant_name>.csv` with the following columns:

| Column | Description |
|---|---|
| `filename` | Image filename (encodes age, gender, race, timestamp) |
| `rating_1` | Rating from first presentation |
| `rating_2` | Rating from second presentation |

---

## Image Dataset

Face images are sourced from the **UTKFace** dataset and organised by age group:

| Age | Google Drive Folder |
|---|---|
| 24 | [Link](https://drive.google.com/drive/folders/1UoLcar2M0fn0XKZQP14PIKQrhzMINW6B) |
| 25 | [Link](https://drive.google.com/drive/folders/124fhUUCG7_87wuODD_Gngyt92oUoM5Q7) |
| 26 | [Link](https://drive.google.com/drive/folders/1aLxAYixMEQ5NsNMlwfoOQ5nLFt4EB8km) |
| 27 | [Link](https://drive.google.com/drive/folders/10tLQrIeekryfSLQaHVEZuhGEtPKhqA4H) |

Filename format: `<age>_<gender>_<race>_<timestamp>.jpg.chip.jpg`
- Gender: `0` = male, `1` = female
- Race: `0` = White, `1` = Black, `2` = Asian, `3` = Indian, `4` = Other

---

## Dependencies

| Package | Purpose |
|---|---|
| `pillow` | Image loading and greyscale conversion |
| `gdown` | Downloading from Google Drive |
| `tkinter` | GUI for the experiment (included with Python) |

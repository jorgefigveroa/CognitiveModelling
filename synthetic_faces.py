from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PCA_DIR = Path("results/pca")
REGRESSION_DIR = Path("results/regression")
OUTPUT_DIR = Path("results/synthetic_faces")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_MODEL_FILE = PCA_DIR / "pca_model.npz"
PCA_SCORES_FILE = PCA_DIR / "pca_scores_48.csv"
LINEAR_MODEL_FILE = REGRESSION_DIR / "linear_model.npz"

REQUESTED_RATINGS = np.arange(0.5, 5.6, 0.5)   # 0.5, 1.0, ..., 5.5


# Helpers
def rating_to_filename(rating: float) -> str:
    return str(rating).replace(".", "p")


def reconstruct_face(
    target_rating: float,
    mean_image: np.ndarray,
    components: np.ndarray,
    selected_pc_indices: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    height: int,
    width: int
) -> np.ndarray:
    """
    Create a synthetic face for a target rating.

    We move along the regression-weight direction in the
    selected PCA subspace and keep all non-selected PCs at 0.
    """

    beta = coefficients
    beta_norm_sq = np.dot(beta, beta)

    if beta_norm_sq == 0:
        raise ValueError("Regression coefficients are all zero.")

    # Scores only in the selected-PC subspace
    selected_scores = ((target_rating - intercept) / beta_norm_sq) * beta

    # Expand to the full 48-PC space
    all_scores = np.zeros(components.shape[0])
    all_scores[selected_pc_indices] = selected_scores

    # Reconstruct image
    face_vector = mean_image + all_scores @ components
    face_image = face_vector.reshape(height, width)

    # Clip to displayable grayscale range
    face_image = np.clip(face_image, 0, 255)

    return face_image.astype(np.uint8)


def save_face(image_array: np.ndarray, path: Path) -> None:
    img = Image.fromarray(image_array, mode="L")
    img.save(path)


def make_contact_sheet(image_paths: list[Path], labels: list[str], output_path: Path) -> None:
    images = [Image.open(path).convert("L") for path in image_paths]

    if not images:
        return

    width, height = images[0].size
    label_height = 30
    padding = 10

    sheet_width = len(images) * (width + padding) + padding
    sheet_height = height + label_height + 2 * padding

    sheet = Image.new("L", (sheet_width, sheet_height), color=255)
    draw = ImageDraw.Draw(sheet)

    for i, (img, label) in enumerate(zip(images, labels)):
        x = padding + i * (width + padding)
        y = padding
        sheet.paste(img, (x, y))
        draw.text((x + 5, y + height + 5), label, fill=0)

    sheet.save(output_path)

# 1. Load saved PCA and regression outputs

pca_data = np.load(PCA_MODEL_FILE)
mean_image = pca_data["mean_image"]
components = pca_data["components"]           # shape: (48, n_pixels)
image_size = pca_data["image_size"]

width = int(image_size[0])
height = int(image_size[1])

linear_model = np.load(LINEAR_MODEL_FILE)
selected_pc_indices = linear_model["selected_pc_indices"].astype(int)
coefficients = linear_model["coefficients"]
intercept = float(linear_model["intercept"][0])

pca_scores_df = pd.read_csv(PCA_SCORES_FILE)
pc_columns = [col for col in pca_scores_df.columns if col.startswith("PC")]

X_all = pca_scores_df[pc_columns].to_numpy()
X_selected = X_all[:, selected_pc_indices]

# 2. Predict ratings for original images
predicted_original = intercept + X_selected @ coefficients

pred_min = predicted_original.min()
pred_max = predicted_original.max()

print(f"Predicted rating range for original faces:")
print(f"Minimum: {pred_min:.4f}")
print(f"Maximum: {pred_max:.4f}")

prediction_df = pd.DataFrame({
    "filename": pca_scores_df["filename"],
    "predicted_rating": predicted_original
})

prediction_df.to_csv(
    OUTPUT_DIR / "predicted_ratings_original_faces.csv",
    index=False
)

# 3. Generate requested synthetic faces
requested_dir = OUTPUT_DIR / "requested_0p5_to_5p5"
requested_dir.mkdir(exist_ok=True)

requested_paths = []
requested_labels = []

for rating in REQUESTED_RATINGS:
    face = reconstruct_face(
        target_rating=rating,
        mean_image=mean_image,
        components=components,
        selected_pc_indices=selected_pc_indices,
        coefficients=coefficients,
        intercept=intercept,
        height=height,
        width=width
    )

    output_path = requested_dir / f"face_rating_{rating_to_filename(rating)}.png"
    save_face(face, output_path)

    requested_paths.append(output_path)
    requested_labels.append(str(rating))

make_contact_sheet(
    requested_paths,
    requested_labels,
    OUTPUT_DIR / "requested_continuum_contact_sheet.png"
)

# 4. Check whether requested ratings are extrapolations
outside_range = (
    REQUESTED_RATINGS.min() < pred_min
    or REQUESTED_RATINGS.max() > pred_max
)

if outside_range:
    print("\nRequested range 0.5-5.5 goes outside the model's observed range.")
    print("Generating an additional 11-face continuum within the observed range.")

    adjusted_ratings = np.linspace(pred_min, pred_max, 11)

    adjusted_dir = OUTPUT_DIR / "within_observed_range"
    adjusted_dir.mkdir(exist_ok=True)

    adjusted_paths = []
    adjusted_labels = []

    for rating in adjusted_ratings:
        face = reconstruct_face(
            target_rating=rating,
            mean_image=mean_image,
            components=components,
            selected_pc_indices=selected_pc_indices,
            coefficients=coefficients,
            intercept=intercept,
            height=height,
            width=width
        )

        output_path = adjusted_dir / f"face_rating_{rating_to_filename(round(float(rating), 3))}.png"
        save_face(face, output_path)

        adjusted_paths.append(output_path)
        adjusted_labels.append(f"{rating:.2f}")

    make_contact_sheet(
        adjusted_paths,
        adjusted_labels,
        OUTPUT_DIR / "within_range_continuum_contact_sheet.png"
    )

    pd.DataFrame({
        "continuum_position": range(1, 12),
        "target_rating": adjusted_ratings
    }).to_csv(
        OUTPUT_DIR / "within_range_target_ratings.csv",
        index=False
    )

else:
    print("\nRequested range 0.5-5.5 is within the model's observed range.")
    adjusted_ratings = REQUESTED_RATINGS

# 5. Save summary
summary = pd.DataFrame({
    "metric": [
        "intercept",
        "number_of_selected_pcs",
        "predicted_min_original_faces",
        "predicted_max_original_faces",
        "requested_min",
        "requested_max",
        "requested_range_outside_observed"
    ],
    "value": [
        intercept,
        len(selected_pc_indices),
        pred_min,
        pred_max,
        REQUESTED_RATINGS.min(),
        REQUESTED_RATINGS.max(),
        outside_range
    ]
})

summary.to_csv(
    OUTPUT_DIR / "synthetic_faces_summary.csv",
    index=False
)

print(f"\nSynthetic face results saved to: {OUTPUT_DIR.resolve()}")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

PCA_DIR = Path("results/pca")
REGRESSION_DIR = Path("results/regression")
OUTPUT_DIR = Path("results/synthetic_faces_6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ratings used to generate the original synthetic stimuli (step 5)
ORIGINAL_STIMULI_RATINGS = np.arange(0.5, 5.6, 0.5)  # 0.5, 1.0, ..., 5.5

# --------------------------------------------------
# Load models
# --------------------------------------------------

pca_data = np.load(PCA_DIR / "pca_model.npz")
mean_image = pca_data["mean_image"]
components = pca_data["components"]   # (48, n_pixels)
image_size = pca_data["image_size"]
width = int(image_size[0])
height = int(image_size[1])

linear_model = np.load(REGRESSION_DIR / "linear_model.npz")
selected_pc_indices = linear_model["selected_pc_indices"].astype(int)
coefficients = linear_model["coefficients"]
intercept = float(linear_model["intercept"][0])

pca_scores_df = pd.read_csv(PCA_DIR / "pca_scores_48.csv")
pc_columns = [col for col in pca_scores_df.columns if col.startswith("PC")]
X_all = pca_scores_df[pc_columns].to_numpy()
X_selected = X_all[:, selected_pc_indices]

# --------------------------------------------------
# 1. Predict ratings for every training stimulus
# --------------------------------------------------

predicted_ratings = intercept + X_selected @ coefficients

pred_min = predicted_ratings.min()
pred_max = predicted_ratings.max()

requested_min = ORIGINAL_STIMULI_RATINGS.min()
requested_max = ORIGINAL_STIMULI_RATINGS.max()

print("=== Point 6: Distribution check ===")
print(f"Predicted rating range for training stimuli:")
print(f"  Minimum : {pred_min:.4f}")
print(f"  Maximum : {pred_max:.4f}")
print()
print(f"Range used to create original synthetic stimuli:")
print(f"  Minimum : {requested_min:.1f}")
print(f"  Maximum : {requested_max:.1f}")
print()

low_deviation  = pred_min - requested_min   # positive means requested goes below distribution
high_deviation = requested_max - pred_max   # positive means requested goes above distribution

print(f"Deviation at low end  : {low_deviation:+.4f}  "
      f"({'outside' if low_deviation > 0 else 'inside'} distribution)")
print(f"Deviation at high end : {high_deviation:+.4f}  "
      f"({'outside' if high_deviation > 0 else 'inside'} distribution)")

# --------------------------------------------------
# 2. Plot distribution of predicted ratings
# --------------------------------------------------

plt.figure(figsize=(7, 4))
plt.hist(predicted_ratings, bins=20, edgecolor="black")
plt.axvline(pred_min, color="red",  linestyle="--", label=f"Predicted min = {pred_min:.2f}")
plt.axvline(pred_max, color="blue", linestyle="--", label=f"Predicted max = {pred_max:.2f}")
plt.axvline(requested_min, color="red",  linestyle=":",
            label=f"Stimuli min = {requested_min:.1f}")
plt.axvline(requested_max, color="blue", linestyle=":",
            label=f"Stimuli max = {requested_max:.1f}")
plt.xlabel("Predicted rating")
plt.ylabel("Number of images")
plt.title("Distribution of model-predicted ratings for training images")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predicted_rating_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved predicted_rating_distribution.png")

# --------------------------------------------------
# 3. Decide whether to generate new faces
# --------------------------------------------------

substantially_outside = (abs(low_deviation) > 0.1) or (abs(high_deviation) > 0.1)

if not substantially_outside:
    print("\nRequested range is within the distribution. No new faces needed.")
else:
    print(f"\nRequested range deviates substantially from the distribution.")
    print("Generating new synthetic faces within the predicted rating range.")

    # --------------------------------------------------
    # 4. Generate 11 new synthetic faces within range
    # --------------------------------------------------

    new_ratings = np.linspace(pred_min, pred_max, 11)

    def reconstruct_face(target_rating):
        beta_norm_sq = np.dot(coefficients, coefficients)
        selected_scores = ((target_rating - intercept) / beta_norm_sq) * coefficients
        all_scores = np.zeros(components.shape[0])
        all_scores[selected_pc_indices] = selected_scores
        face_vector = mean_image + all_scores @ components
        face_image = np.clip(face_vector.reshape(height, width), 0, 255)
        return face_image.astype(np.uint8)

    faces_dir = OUTPUT_DIR / "faces"
    faces_dir.mkdir(exist_ok=True)

    face_paths = []
    face_labels = []

    for rating in new_ratings:
        face = reconstruct_face(rating)
        label = f"{rating:.2f}"
        filename = f"face_{label.replace('.', 'p')}.png"
        path = faces_dir / filename
        Image.fromarray(face, mode="L").save(path)
        face_paths.append(path)
        face_labels.append(label)

    print(f"Saved {len(face_paths)} faces to {faces_dir}")

    # Save filename → target rating map for use by analysis scripts
    pd.DataFrame({
        "filename": [p.name for p in face_paths],
        "target_rating": new_ratings
    }).to_csv(OUTPUT_DIR / "face_target_ratings.csv", index=False)

    # --------------------------------------------------
    # 5. Contact sheet
    # --------------------------------------------------

    images = [Image.open(p).convert("L") for p in face_paths]
    img_w, img_h = images[0].size
    label_h = 30
    padding = 10

    sheet_w = len(images) * (img_w + padding) + padding
    sheet_h = img_h + label_h + 2 * padding
    sheet = Image.new("L", (sheet_w, sheet_h), color=255)
    draw = ImageDraw.Draw(sheet)

    for i, (img, label) in enumerate(zip(images, face_labels)):
        x = padding + i * (img_w + padding)
        sheet.paste(img, (x, padding))
        draw.text((x + 5, padding + img_h + 5), label, fill=0)

    sheet_path = OUTPUT_DIR / "contact_sheet_within_range.png"
    sheet.save(sheet_path)
    print(f"Saved contact sheet to {sheet_path}")

    # --------------------------------------------------
    # 6. Save summary table
    # --------------------------------------------------

    summary = pd.DataFrame({
        "metric": [
            "predicted_min_training",
            "predicted_max_training",
            "original_stimuli_min",
            "original_stimuli_max",
            "deviation_low_end",
            "deviation_high_end",
            "new_faces_generated",
        ],
        "value": [
            pred_min,
            pred_max,
            requested_min,
            requested_max,
            low_deviation,
            high_deviation,
            len(new_ratings),
        ]
    })

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\nSummary saved to {OUTPUT_DIR / 'summary.csv'}")
    print("\n=== Summary for report ===")
    print(summary.to_string(index=False))

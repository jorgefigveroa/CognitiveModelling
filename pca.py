
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

# Folder containing all grayscale images
IMAGE_DIR = Path("images/greyscale")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
RESULTS_DIR = Path("results/pca")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load all grayscale images
image_paths = sorted([
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
])

print(f"Found {len(image_paths)} images")

images = []
filenames = []

image_size = None

for image_path in image_paths:

    with Image.open(image_path) as img:

        img = img.convert("L")

        # Save expected image size from first image
        if image_size is None:
            image_size = img.size

        # PCA requires all images to have same dimensions
        elif img.size != image_size:
            raise ValueError(
                f"{image_path.name} has size {img.size}, "
                f"but expected {image_size}"
            )

        img_array = np.asarray(img, dtype=np.float64)

        # Flatten image into one long vector
        images.append(img_array.flatten())

        filenames.append(image_path.name)


# Stack all images into matrix J
J = np.array(images)

print("Image size:", image_size)
print("J shape:", J.shape)

# 2. Calculate average image and centre data
mean_image = J.mean(axis=0)
J_centered = J - mean_image

print(
    "Largest absolute mean after centering:",
    np.abs(J_centered.mean(axis=0)).max()
)

# 3. Show average face
width, height = image_size

plt.imshow(
    mean_image.reshape(height, width),
    cmap="gray"
)

plt.title("Average face")
plt.axis("off")

plt.savefig(
    RESULTS_DIR / "average_face.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 4. Run PCA
pca = PCA()
scores = pca.fit_transform(J_centered)

print("Scores shape:", scores.shape)
print("Components shape:", pca.components_.shape)

# 5. Plot variance explained by each PC
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 5))

plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Proportion of variance explained")
plt.title("Variance explained by each principal component")

plt.savefig(
    RESULTS_DIR / "variance_explained.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 6. Plot cumulative variance explained
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.xlabel("Number of principal components")
plt.ylabel("Cumulative variance explained")
plt.title("Cumulative variance explained")

plt.grid()

plt.savefig(
    RESULTS_DIR / "cumulative_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Show number of PCs needed for common thresholds
for threshold in [0.80, 0.90, 0.95]:

    n_components = np.argmax(
        cumulative_variance >= threshold
    ) + 1

    print(
        f"{threshold:.0%} variance explained by "
        f"{n_components} PCs"
    )

# 7. Visualise first few principal components
number_of_pcs_to_show = 5
for pc_index in range(number_of_pcs_to_show):

    pc = pca.components_[pc_index]

    min_score = scores[:, pc_index].min()
    max_score = scores[:, pc_index].max()

    min_face = mean_image + min_score * pc
    max_face = mean_image + max_score * pc

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(
        min_face.reshape(height, width),
        cmap="gray"
    )
    axes[0].set_title("Minimum score")

    axes[1].imshow(
        mean_image.reshape(height, width),
        cmap="gray"
    )
    axes[1].set_title("Average face")

    axes[2].imshow(
        max_face.reshape(height, width),
        cmap="gray"
    )
    axes[2].set_title("Maximum score")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"Principal Component {pc_index + 1}"
    )

    plt.tight_layout()

    plt.savefig(
    RESULTS_DIR / f"PC{pc_index + 1}.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()
    
import pandas as pd
N_PCS_FOR_MODEL = 48

pca_scores = pd.DataFrame(
    scores[:, :N_PCS_FOR_MODEL],
    columns=[f"PC{i + 1}" for i in range(N_PCS_FOR_MODEL)]
)

pca_scores.insert(0, "filename", filenames)

pca_scores.to_csv(
    RESULTS_DIR / "pca_scores_48.csv",
    index=False
)

np.savez(
    RESULTS_DIR / "pca_model.npz",
    mean_image=mean_image,
    components=pca.components_[:N_PCS_FOR_MODEL],
    image_size=np.array([width, height])
)

print("Saved PCA scores and model.")
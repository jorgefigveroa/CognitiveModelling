from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# Settings
# --------------------------------------------------

RESULTS_DIR = Path("results")
PCA_DIR = RESULTS_DIR / "pca"
REGRESSION_DIR = RESULTS_DIR / "regression"

REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

RATING_FILES = {
    "Ania": RESULTS_DIR / "Ania.csv",
    "Jorge": RESULTS_DIR / "Jorge.csv",
    "Maria": RESULTS_DIR / "maria.csv",
}

PCA_SCORES_FILE = PCA_DIR / "pca_scores_48.csv"

N_SPLITS = 5


# --------------------------------------------------
# 1. Load PCA scores
# --------------------------------------------------

pca_scores = pd.read_csv(PCA_SCORES_FILE)

pca_scores["filename"] = (
    pca_scores["filename"]
    .astype(str)
    .map(lambda x: Path(x.strip()).name)
)

pc_columns = [
    column
    for column in pca_scores.columns
    if column.startswith("PC")
]

print(f"Loaded {len(pca_scores)} images")
print(f"Using {len(pc_columns)} candidate PCs")


# --------------------------------------------------
# 2. Load participant ratings
# --------------------------------------------------

all_ratings = []

for participant, csv_path in RATING_FILES.items():

    df = pd.read_csv(csv_path)

    required_columns = {"filename", "rating_1", "rating_2"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain "
            f"filename, rating_1 and rating_2"
        )

    df["filename"] = (
        df["filename"]
        .astype(str)
        .map(lambda x: Path(x.strip()).name)
    )

    ratings_raw = df[["rating_1", "rating_2"]].to_numpy().flatten()

    rating_min = ratings_raw.min()
    rating_max = ratings_raw.max()

    print(
        f"{participant}: "
        f"min={rating_min}, max={rating_max}, "
        f"n={len(ratings_raw)}"
    )

    # --------------------------------------------------
    # Normalise only if participant did not use
    # the full 1-5 scale
    # --------------------------------------------------

    if rating_min > 1 or rating_max < 5:

        print(
            f"  -> {participant} did not use the full scale. "
            "Applying min-max normalisation to 1-5."
        )

        if rating_max == rating_min:
            raise ValueError(
                f"{participant} used only one rating value; "
                "cannot min-max normalise."
            )

        for column in ["rating_1", "rating_2"]:

            df[column] = (
                1
                + 4
                * (df[column] - rating_min)
                / (rating_max - rating_min)
            )

    else:
        print(
            f"  -> {participant} used the full 1-5 scale. "
            "No normalisation applied."
        )

    # Convert from:
    #
    # filename | rating_1 | rating_2
    #
    # to:
    #
    # filename | repetition | rating
    #
    long_df = df.melt(
        id_vars="filename",
        value_vars=["rating_1", "rating_2"],
        var_name="repetition",
        value_name="rating"
    )

    long_df["participant"] = participant

    all_ratings.append(long_df)


ratings = pd.concat(
    all_ratings,
    ignore_index=True
)

print(f"\nTotal rating observations: {len(ratings)}")


# --------------------------------------------------
# 3. Plot raw/processed rating histograms
# --------------------------------------------------

for participant in RATING_FILES:

    participant_ratings = ratings.loc[
        ratings["participant"] == participant,
        "rating"
    ]

    plt.figure(figsize=(6, 4))

    plt.hist(
        participant_ratings,
        bins=np.arange(0.5, 6, 1),
        edgecolor="black"
    )

    plt.xticks([1, 2, 3, 4, 5])

    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title(f"Rating distribution - {participant}")

    plt.savefig(
        REGRESSION_DIR / f"ratings_{participant}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 4. Match ratings to PCA scores
# --------------------------------------------------

rating_filenames = set(ratings["filename"])
pca_filenames = set(pca_scores["filename"])

missing_from_pca = rating_filenames - pca_filenames
missing_from_ratings = pca_filenames - rating_filenames

if missing_from_pca:
    print("\nRatings with no matching PCA image:")
    for name in sorted(missing_from_pca):
        print(name)

if missing_from_ratings:
    print("\nPCA images with no rating:")
    for name in sorted(missing_from_ratings):
        print(name)


data = ratings.merge(
    pca_scores,
    on="filename",
    how="inner",
    validate="many_to_one"
)

print(f"\nMatched observations: {len(data)}")
print(f"Unique images matched: {data['filename'].nunique()}")


# --------------------------------------------------
# 5. Prepare regression variables
# --------------------------------------------------

X = data[pc_columns].to_numpy()
y = data["rating"].to_numpy()

# Important:
# group by filename so ratings of the same face
# cannot appear in both training and validation data.
groups = data["filename"].to_numpy()


# --------------------------------------------------
# 6. Cross-validation helpers
# --------------------------------------------------

def baseline_cv_mse(y, groups):
    """
    Cross-validated MSE for a model that predicts only
    the training-set mean rating.
    """

    cv = GroupKFold(n_splits=N_SPLITS)

    fold_errors = []

    dummy_x = np.zeros((len(y), 1))

    for train_index, test_index in cv.split(
        dummy_x,
        y,
        groups
    ):

        train_mean = y[train_index].mean()

        prediction = np.full(
            len(test_index),
            train_mean
        )

        mse = mean_squared_error(
            y[test_index],
            prediction
        )

        fold_errors.append(mse)

    return np.mean(fold_errors)


def cv_mse_for_features(feature_indices):
    """
    Cross-validated MSE for a given collection of PCs.
    """

    cv = GroupKFold(n_splits=N_SPLITS)

    fold_errors = []

    X_selected = X[:, feature_indices]

    for train_index, test_index in cv.split(
        X_selected,
        y,
        groups
    ):

        model = LinearRegression()

        model.fit(
            X_selected[train_index],
            y[train_index]
        )

        prediction = model.predict(
            X_selected[test_index]
        )

        mse = mean_squared_error(
            y[test_index],
            prediction
        )

        fold_errors.append(mse)

    return np.mean(fold_errors)


# --------------------------------------------------
# 7. Forward selection
# --------------------------------------------------

selected = []
remaining = list(range(len(pc_columns)))

current_mse = baseline_cv_mse(
    y,
    groups
)

print(
    f"\nBaseline cross-validated MSE: "
    f"{current_mse:.4f}"
)

history = []

while remaining:

    best_feature = None
    best_mse = np.inf

    for candidate in remaining:

        candidate_features = selected + [candidate]

        candidate_mse = cv_mse_for_features(
            candidate_features
        )

        if candidate_mse < best_mse:

            best_mse = candidate_mse
            best_feature = candidate

    improvement = current_mse - best_mse

    # Stop when adding another PC no longer
    # improves cross-validated prediction.
    if improvement <= 1e-8:

        print(
            "\nNo remaining PC improves "
            "cross-validated performance."
        )

        break

    selected.append(best_feature)
    remaining.remove(best_feature)

    current_mse = best_mse

    selected_name = pc_columns[best_feature]

    history.append({
        "step": len(selected),
        "selected_pc": selected_name,
        "cv_mse": current_mse,
        "cv_rmse": np.sqrt(current_mse)
    })

    print(
        f"Step {len(selected)}: "
        f"selected {selected_name}, "
        f"CV RMSE = {np.sqrt(current_mse):.4f}"
    )


selected_pc_names = [
    pc_columns[index]
    for index in selected
]

print("\nSelected PCs:")
print(selected_pc_names)


# --------------------------------------------------
# 8. Fit final model to all observations
# --------------------------------------------------

X_final = X[:, selected]

final_model = LinearRegression()

final_model.fit(
    X_final,
    y
)

fitted_ratings = final_model.predict(
    X_final
)

training_rmse = np.sqrt(
    mean_squared_error(
        y,
        fitted_ratings
    )
)

training_r2 = r2_score(
    y,
    fitted_ratings
)

print(
    f"\nFinal model intercept: "
    f"{final_model.intercept_:.4f}"
)

print(
    f"Training RMSE: "
    f"{training_rmse:.4f}"
)

print(
    f"Training R²: "
    f"{training_r2:.4f}"
)


# --------------------------------------------------
# 9. Save forward-selection results
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    REGRESSION_DIR / "forward_selection.csv",
    index=False
)


coefficients = pd.DataFrame({
    "PC": selected_pc_names,
    "coefficient": final_model.coef_
})

coefficients.to_csv(
    REGRESSION_DIR / "selected_pcs.csv",
    index=False
)


# Save matched data for transparency/checking
data.to_csv(
    REGRESSION_DIR / "model_data.csv",
    index=False
)


# --------------------------------------------------
# 10. Save model for synthetic-face generation
# --------------------------------------------------

np.savez(
    REGRESSION_DIR / "linear_model.npz",
    selected_pc_indices=np.array(selected),
    coefficients=final_model.coef_,
    intercept=np.array([final_model.intercept_])
)


print(
    f"\nRegression results saved to "
    f"{REGRESSION_DIR.resolve()}"
)
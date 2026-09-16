from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

EXPERIMENT_2_DIR = Path("results/experiment_2")
FACE_MAP_FILE    = Path("results/synthetic_faces_6/face_target_ratings.csv")
OUTPUT_DIR       = Path("results/synthetic_faces_7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load filename → target rating map
# --------------------------------------------------

face_map = pd.read_csv(FACE_MAP_FILE)
face_map = face_map.set_index("filename")["target_rating"].to_dict()

# --------------------------------------------------
# 2. Load all participant results
# --------------------------------------------------

result_files = sorted(EXPERIMENT_2_DIR.glob("*.csv"))

if not result_files:
    raise FileNotFoundError(
        f"No experiment-2 CSV files found in {EXPERIMENT_2_DIR}.\n"
        "Run experiment_2.py for each participant first."
    )

participants = []
all_data = []

for path in result_files:
    df = pd.read_csv(path)
    participant = path.stem
    df["participant"] = participant
    df["target_rating"] = df["filename"].map(face_map)

    missing = df["target_rating"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows in {path.name} could not be matched to a face.")
        df = df.dropna(subset=["target_rating"])

    all_data.append(df)
    participants.append(participant)
    print(f"Loaded {path.name}: {len(df)} trials, {df['filename'].nunique()} unique faces")

data = pd.concat(all_data, ignore_index=True)

print(f"\nTotal trials: {len(data)}")
print(f"Participants: {participants}")

# --------------------------------------------------
# 3. Box plots — one subplot per participant
# --------------------------------------------------

n_participants = len(participants)
fig, axes = plt.subplots(
    1, n_participants,
    figsize=(5 * n_participants, 5),
    sharey=True
)

if n_participants == 1:
    axes = [axes]

spearman_results = {}

for ax, participant in zip(axes, participants):

    pdata = data[data["participant"] == participant].copy()

    # Groups for box plot: sorted unique predicted ratings
    sorted_ratings = sorted(pdata["target_rating"].unique())
    groups = [
        pdata.loc[pdata["target_rating"] == r, "rating"].values
        for r in sorted_ratings
    ]

    ax.boxplot(
        groups,
        positions=range(len(sorted_ratings)),
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue"),
        medianprops=dict(color="navy", linewidth=2),
    )

    # Spearman's ρ between each trial's predicted rating and actual rating
    rho, pval = spearmanr(pdata["target_rating"], pdata["rating"])
    spearman_results[participant] = (rho, pval)

    ax.set_xticks(range(len(sorted_ratings)))
    ax.set_xticklabels(
        [f"{r:.2f}" for r in sorted_ratings],
        rotation=45,
        ha="right",
        fontsize=8
    )
    ax.set_xlabel("Predicted rating")
    ax.set_ylabel("Participant rating (1–5)")
    ax.set_title(f"{participant}\nSpearman ρ = {rho:.3f}  (p = {pval:.3f})")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])

plt.suptitle(
    "Experiment 2: Participant ratings vs model-predicted rating",
    fontsize=13,
    y=1.02
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplots.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved boxplots.png to {OUTPUT_DIR}")

# --------------------------------------------------
# 4. Print Spearman summary
# --------------------------------------------------

print("\n=== Spearman rank correlation (ρ) ===")
print(f"{'Participant':<20} {'ρ':>8} {'p-value':>10}  {'Close to 1?':>12}")
print("-" * 55)

for participant, (rho, pval) in spearman_results.items():
    close = "Yes" if rho >= 0.7 else "No"
    print(f"{participant:<20} {rho:>8.3f} {pval:>10.4f}  {close:>12}")

# --------------------------------------------------
# 5. Save Spearman results to CSV
# --------------------------------------------------

summary_df = pd.DataFrame([
    {
        "participant": p,
        "spearman_rho": rho,
        "p_value": pval,
        "close_to_1": rho >= 0.7
    }
    for p, (rho, pval) in spearman_results.items()
])

summary_df.to_csv(OUTPUT_DIR / "spearman_results.csv", index=False)
print(f"\nSpearman results saved to {OUTPUT_DIR / 'spearman_results.csv'}")

# --------------------------------------------------
# 6. Mean rating per face across all participants (overview plot)
# --------------------------------------------------

mean_per_face = (
    data.groupby("target_rating")["rating"]
    .agg(["mean", "sem"])
    .reset_index()
)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.errorbar(
    mean_per_face["target_rating"],
    mean_per_face["mean"],
    yerr=mean_per_face["sem"],
    fmt="o-",
    capsize=4,
    color="steelblue"
)
ax2.plot(
    [mean_per_face["target_rating"].min(), mean_per_face["target_rating"].max()],
    [mean_per_face["target_rating"].min(), mean_per_face["target_rating"].max()],
    "k--",
    alpha=0.4,
    label="Perfect correlation"
)
ax2.set_xlabel("Predicted rating")
ax2.set_ylabel("Mean participant rating (± SEM)")
ax2.set_title("Mean rating vs predicted rating (all participants)")
ax2.legend()
ax2.set_ylim(0.5, 5.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_rating_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved mean_rating_vs_predicted.png to {OUTPUT_DIR}")

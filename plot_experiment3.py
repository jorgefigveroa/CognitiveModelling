import os
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Folder containing Experiment 3 results
# --------------------------------------------------

results_folder = "results/experiment3"

# File containing information about the selected faces
faces_file = os.path.join(
    results_folder,
    "experiment3_faces.csv"
)


# --------------------------------------------------
# Load information about the selected faces
# --------------------------------------------------

faces = pd.read_csv(faces_file)


# Find low and high adapting faces
low_adapter = faces[
    faces["type"] == "adapter_low"
]["filename"].iloc[0]

high_adapter = faces[
    faces["type"] == "adapter_high"
]["filename"].iloc[0]


# Find the three test faces
test_faces = faces[
    faces["type"] == "test"
]["filename"].tolist()


print("Low adapter:", low_adapter)
print("High adapter:", high_adapter)

print("\nTest faces:")
print(test_faces)


# --------------------------------------------------
# Find participant result files
# --------------------------------------------------

csv_files = []

for filename in os.listdir(results_folder):

    if filename.endswith(".csv"):

        # Do not include experiment3_faces.csv
        if filename != "experiment3_faces.csv":

            csv_files.append(filename)


print("\nParticipant files found:")
print(csv_files)


# --------------------------------------------------
# Read all participant results
# --------------------------------------------------

all_results = []


for filename in csv_files:

    path = os.path.join(
        results_folder,
        filename
    )

    data = pd.read_csv(path)

    # Participant name = filename without .csv
    participant = filename.replace(".csv", "")

    data["participant"] = participant

    all_results.append(data)


# Combine all participants into one table
results = pd.concat(
    all_results,
    ignore_index=True
)


# --------------------------------------------------
# Add adaptation condition
# --------------------------------------------------

def get_condition(adapting_face):

    if adapting_face == low_adapter:
        return "Low adapter"

    elif adapting_face == high_adapter:
        return "High adapter"

    else:
        return "Unknown"


results["condition"] = results[
    "adapting_face"
].apply(get_condition)


# --------------------------------------------------
# Calculate mean ratings
# --------------------------------------------------

mean_ratings = results.groupby(
    ["test_face", "condition"]
)["rating"].mean().reset_index()


print("\nMean ratings:")
print(mean_ratings)


# --------------------------------------------------
# Prepare data for plot
# --------------------------------------------------

low_means = []
high_means = []


for test_face in test_faces:

    # Mean after low adapter
    low_value = mean_ratings[
        (mean_ratings["test_face"] == test_face)
        &
        (mean_ratings["condition"] == "Low adapter image")
    ]["rating"].iloc[0]

    low_means.append(low_value)


    # Mean after high adapter
    high_value = mean_ratings[
        (mean_ratings["test_face"] == test_face)
        &
        (mean_ratings["condition"] == "High adapter image")
    ]["rating"].iloc[0]

    high_means.append(high_value)


# --------------------------------------------------
# Create grouped bar plot
# --------------------------------------------------

x = range(len(test_faces))

bar_width = 0.35


plt.figure(figsize=(8, 5))


plt.bar(
    [i - bar_width / 2 for i in x],
    low_means,
    width=bar_width,
    label="Low adapter"
)


plt.bar(
    [i + bar_width / 2 for i in x],
    high_means,
    width=bar_width,
    label="High adapter"
)


# --------------------------------------------------
# Labels
# --------------------------------------------------

plt.xlabel("Test stimulus")

plt.ylabel("Mean rating")

plt.title(
    "Experiment 3: Perceptual after-effect"
)


plt.xticks(
    x,
    ["Test 1", "Test 2", "Test 3"]
)


# Rating scale is 1 to 5
plt.ylim(1, 5)

plt.yticks([1, 2, 3, 4, 5])

plt.legend()

plt.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

output_path = os.path.join(
    results_folder,
    "experiment3_results.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nPlot saved to:")
print(output_path)
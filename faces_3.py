import os
import pandas as pd




ratings_file = "results/synthetic_faces_6/face_target_ratings.csv"

output_folder = "results/experiment3"

os.makedirs(output_folder, exist_ok=True)

# File containing the 5 faces selected for Experiment 3
output_file = os.path.join(
    output_folder,
    "experiment3_faces.csv"
)



faces = pd.read_csv(ratings_file)

faces = faces.sort_values("target_rating").reset_index(drop=True)

print("Synthetic faces from Part 6:")
print(faces)




# First face = lowest rating
low_face = faces.iloc[0]

# Last face = highest rating
high_face = faces.iloc[-1]



minimum = faces["target_rating"].min()
maximum = faces["target_rating"].max()

middle = (minimum + maximum) / 2

print("\nMinimum rating:", minimum)
print("Maximum rating:", maximum)
print("Middle rating:", middle)



# calculated the distance of each face from the middle
faces["distance"] = abs(
    faces["target_rating"] - middle
)

# we sorted by the distances and took the 3 closest faces 
middle_faces = faces.sort_values("distance").head(3)

# put the three faces in orders
middle_faces = middle_faces.sort_values("target_rating")



selected_faces = []


# Add low endpoint
selected_faces.append([
    "adapter_low",
    low_face["filename"],
    low_face["target_rating"]
])


# Add the three test faces
for index, face in middle_faces.iterrows():

    selected_faces.append([
        "test",
        face["filename"],
        face["target_rating"]
    ])


# Add high endpoint
selected_faces.append([
    "adapter_high",
    high_face["filename"],
    high_face["target_rating"]
])


selected_faces = pd.DataFrame(
    selected_faces,
    columns=[
        "type",
        "filename",
        "target_rating"
    ]
)
#saved the faces to a file for experiment 3

selected_faces.to_csv(
    output_file,
    index=False
)

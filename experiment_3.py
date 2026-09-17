import os
import csv
import random
import tkinter as tk
import pandas as pd
from PIL import Image, ImageTk, ImageDraw




# Folder containing synthetic faces from Part 6
faces_folder = "results/synthetic_faces_6/faces"

# CSV created by faces_3.py
selected_faces_file = "results/experiment3/experiment3_faces.csv"

# Folder where participant results will be saved
results_folder = "results/experiment3"

os.makedirs(results_folder, exist_ok=True)


participant_id = input("Enter student name: ")



faces = pd.read_csv(selected_faces_file)


# Get the two adapting faces
adapting_faces = faces[
    faces["type"].isin(["adapter_low", "adapter_high"])
]["filename"].tolist()


# Get the three test faces
test_faces = faces[
    faces["type"] == "test"
]["filename"].tolist()


print("\nAdapting faces:")
print(adapting_faces)

print("\nTest faces:")
print(test_faces)


# --------------------------------------------------
# Create the 6 experimental conditions
# --------------------------------------------------

trials = []

for adapting_face in adapting_faces:

    for test_face in test_faces:

        trials.append([
            adapting_face,
            test_face
        ])


# Randomize the order
random.shuffle(trials)

print("\nNumber of trials:", len(trials))




results = []


# --------------------------------------------------
# Create experiment window
# --------------------------------------------------

root = tk.Tk()

root.title("Experiment 3")

root.configure(bg="white")
root.geometry("1400x950")


image_label = tk.Label(root, bg="white")
image_label.pack(expand=True)


instruction_label = tk.Label(
    root,
    text="",
    font=("Arial", 18),
    bg="white",
    fg="black"
)
instruction_label.pack(pady=10)


progress_label = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    bg="white",
    fg="black"
)
progress_label.pack(pady=10)




current_trial = 0
current_photo = None
waiting_for_rating = False




def show_image(filename):

    global current_photo

    # Find image
    path = os.path.join(
        faces_folder,
        filename
    )

    # Open image
    image = Image.open(path)

    # Convert to RGB so we can add a fixation cross
    image = image.convert("RGB")

    # Resize image if needed
    image.thumbnail((1300, 800))


    draw = ImageDraw.Draw(image)

    width, height = image.size

    center_x = width // 2
    center_y = height // 2


    # Horizontal line
    draw.line(
        (
            center_x - 8,
            center_y,
            center_x + 8,
            center_y
        ),
        fill="red",
        width=2
    )


    # Vertical line
    draw.line(
        (
            center_x,
            center_y - 8,
            center_x,
            center_y + 8
        ),
        fill="red",
        width=2
    )


    # Convert image for Tkinter
    current_photo = ImageTk.PhotoImage(image)

    # Display image
    image_label.config(image=current_photo)


# --------------------------------------------------
# Start one trial
# --------------------------------------------------

def start_trial():

    global waiting_for_rating

    waiting_for_rating = False


    # Get adapting face for current trial
    adapting_face = trials[current_trial][0]


    instruction_label.config(text="Look at the fixation cross")


    progress_label.config(
        text=f"Trial {current_trial + 1} of {len(trials)}"
    )


    # Show adapting face
    show_image(adapting_face)


    # Wait 25 seconds before showing test face
    root.after(
        25000,
        show_test_face
    )



def show_test_face():

    # Get test face for current trial
    test_face = trials[current_trial][1]


    # Show test face
    show_image(test_face)


    # Show it for 750 milliseconds
    root.after(
        750,
        ask_for_rating
    )



def ask_for_rating():

    global waiting_for_rating


    # Remove image
    image_label.config(image="")


    # Ask participant
    instruction_label.config(text="Rate the face from 1 to 5")


    # Now keyboard responses are allowed
    waiting_for_rating = True



def record_rating(event):

    global current_trial
    global waiting_for_rating


 
    if waiting_for_rating == False:
        return


    key = event.char


    # Only accept 1, 2, 3, 4 or 5
    if key not in ["1", "2", "3", "4", "5"]:
        return


    adapting_face = trials[current_trial][0]
    test_face = trials[current_trial][1]

    rating = int(key)


    # Save result
    results.append([
        adapting_face,
        test_face,
        rating
    ])


    waiting_for_rating = False

    current_trial += 1


    # Start next trial
    if current_trial < len(trials):

        start_trial()

    # Or finish experiment
    else:

        finish_experiment()


def finish_experiment():

    output_path = os.path.join(
        results_folder,
        participant_id + ".csv"
    )


    with open(output_path, "w", newline="") as file:

        writer = csv.writer(file)


        # Column names
        writer.writerow([
            "adapting_face",
            "test_face",
            "rating"
        ])


        # Participant results
        for result in results:

            writer.writerow(result)


    # Remove image
    image_label.config(image="")


    # Show final message
    instruction_label.config(text="Experiment complete. Thank you!")


    progress_label.config(
        text=f"Results saved to {output_path}"
    )


    # Stop accepting keyboard responses and close after 3 seconds
    root.unbind("<Key>")
    root.after(3000, root.destroy)



root.bind("<Key>", record_rating)


# --------------------------------------------------
# Start experiment
# --------------------------------------------------

start_trial()

root.mainloop()
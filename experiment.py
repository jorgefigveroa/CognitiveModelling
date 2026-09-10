import os
import csv
import random
import tkinter as tk
from PIL import Image, ImageTk


images_folder="images/greyscale"
results_folder="results"
participant_id = input("Enter student name: ")
os.makedirs(results_folder, exist_ok=True)

images = [
    filename
    for filename in os.listdir(images_folder)
]

print(f"{len(images)} images found.")

trials = images * 2
random.shuffle(trials)

ratings = {}

for image in images:
    ratings[image] = []


root = tk.Tk()
root.title("Image Rating Experiment")

root.geometry("900x700")

image_label = tk.Label(root)
image_label.pack(expand=True)

instruction_label = tk.Label(
    root,
    text="Rate the image from 1 to 5",
    font=("Arial", 18)
)
instruction_label.pack(pady=10)

progress_label = tk.Label(
    root,
    text="",
    font=("Arial", 12)
)
progress_label.pack(pady=10)

current_trial = 0
current_photo = None

def show_image():

    global current_photo

    filename = trials[current_trial]

    path = os.path.join(images_folder, filename)

    image = Image.open(path)

    # Resize image to fit window
    image.thumbnail((800, 550))

    current_photo = ImageTk.PhotoImage(image)

    image_label.config(image=current_photo)

    progress_label.config(
        text=f"Image {current_trial + 1} of {len(trials)}"
    )


def record_rating(event):

    global current_trial

    key = event.char

    if key not in ["1", "2", "3", "4", "5"]:
        return

    filename = trials[current_trial]

    rating = int(key)

    ratings[filename].append(rating)

    current_trial += 1

    if current_trial < len(trials):
        show_image()
    else:
        finish_experiment()


def finish_experiment():

    output_path = os.path.join(
        results_folder,
        participant_id + ".csv"
    )

    with open(output_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "rating_1",
            "rating_2"
        ])

        for filename in images:

            writer.writerow([
                filename,
                ratings[filename][0],
                ratings[filename][1]
            ])

    image_label.config(image="")
    instruction_label.config(
        text="Experiment complete. Thank you!"
    )

    progress_label.config(
        text=f"Results saved to {output_path}"
    )

    root.unbind("<Key>")

    
root.bind("<Key>", record_rating)

show_image()

root.mainloop()
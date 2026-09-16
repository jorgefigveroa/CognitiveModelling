import csv
import random
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

IMAGES_FOLDER = Path("results/synthetic_faces_6/faces")
RESULTS_FOLDER = Path("results/experiment_2")
REPETITIONS = 10   # each synthetic face shown this many times

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

participant_id = input("Enter participant name: ").strip()
if not participant_id:
    raise ValueError("Participant name cannot be empty.")

output_path = RESULTS_FOLDER / f"{participant_id}.csv"

if output_path.exists():
    raise FileExistsError(
        f"Results file already exists for '{participant_id}': {output_path}\n"
        "Choose a different name or delete the existing file first."
    )

image_files = sorted([
    f.name
    for f in IMAGES_FOLDER.iterdir()
    if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
])

if not image_files:
    raise FileNotFoundError(f"No images found in {IMAGES_FOLDER}")

print(f"{len(image_files)} synthetic images found.")
print(f"Each shown {REPETITIONS} times → {len(image_files) * REPETITIONS} trials total.")

# Build trial list: each image repeated REPETITIONS times, then shuffled
trials = image_files * REPETITIONS
random.shuffle(trials)

trial_records = []   # list of (filename, trial_number, rating)
trial_counters = {f: 0 for f in image_files}

root = tk.Tk()
root.title("Experiment 2 – Synthetic Face Rating")
root.geometry("900x700")

image_label = tk.Label(root)
image_label.pack(expand=True)

instruction_label = tk.Label(
    root,
    text="Rate the image from 1 to 5",
    font=("Arial", 18)
)
instruction_label.pack(pady=10)

progress_label = tk.Label(root, text="", font=("Arial", 12))
progress_label.pack(pady=10)

current_trial = 0
current_photo = None


def show_image():
    global current_photo

    filename = trials[current_trial]
    path = IMAGES_FOLDER / filename
    image = Image.open(path)
    image.thumbnail((800, 550))
    current_photo = ImageTk.PhotoImage(image)

    image_label.config(image=current_photo)
    progress_label.config(
        text=f"Trial {current_trial + 1} of {len(trials)}"
    )


def record_rating(event):
    global current_trial

    key = event.char
    if key not in ["1", "2", "3", "4", "5"]:
        return

    filename = trials[current_trial]
    trial_counters[filename] += 1
    trial_records.append((filename, trial_counters[filename], int(key)))

    current_trial += 1
    if current_trial < len(trials):
        show_image()
    else:
        finish_experiment()


def finish_experiment():
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "trial", "rating"])
        writer.writerows(trial_records)

    image_label.config(image="")
    instruction_label.config(text="Experiment complete. Thank you!")
    progress_label.config(text=f"Results saved to {output_path}")
    root.unbind("<Key>")


root.bind("<Key>", record_rating)
show_image()
root.mainloop()

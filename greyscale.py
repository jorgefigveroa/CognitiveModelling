import tempfile
from pathlib import Path
from PIL import Image
import gdown

FOLDERS = {
    "24": "1UoLcar2M0fn0XKZQP14PIKQrhzMINW6B",
    "25": "124fhUUCG7_87wuODD_Gngyt92oUoM5Q7",
    "26": "1aLxAYixMEQ5NsNMlwfoOQ5nLFt4EB8km",
    "27": "10tLQrIeekryfSLQaHVEZuhGEtPKhqA4H",
}
GREY_DIR = Path("images/greyscale")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def process_images(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in input_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        img = Image.open(image_path).convert("L")
        img = img.resize((28, 28))
        img.save(output_dir / image_path.name)
        print(f"Saved: {image_path.name}")


if __name__ == "__main__":
    for age, folder_id in FOLDERS.items():
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        out_dir = GREY_DIR / age
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"Skipping {age} — already processed.")
            continue
        print(f"\nProcessing age {age}...")
        with tempfile.TemporaryDirectory() as tmp:
            gdown.download_folder(url, output=tmp, quiet=False)
            process_images(Path(tmp), out_dir)
        print(f"Age {age} done: {len(list(out_dir.iterdir()))} images saved.")
    print(f"\nAll done. Images in {GREY_DIR}")

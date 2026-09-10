import tempfile
from pathlib import Path
from PIL import Image
import gdown

FOLDER_ID = "1UoLcar2M0fn0XKZQP14PIKQrhzMINW6B"
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
    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    with tempfile.TemporaryDirectory() as tmp:
        gdown.download_folder(url, output=tmp, quiet=False)
        process_images(Path(tmp), GREY_DIR)
    print(f"Done. Grayscale images saved to {GREY_DIR}")

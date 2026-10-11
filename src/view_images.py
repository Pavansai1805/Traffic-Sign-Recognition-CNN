
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = Path(
    r"C:\TrafficSignProject\dataset\archive (1)\Train"
)

# Find all class folders
class_folders = [
    folder for folder in dataset_path.iterdir()
    if folder.is_dir()
]

# Select 9 random images
image_paths = []

for folder in class_folders:
    images = list(folder.glob("*.png"))
    if images:
        image_paths.append(random.choice(images))

random.shuffle(image_paths)
image_paths = image_paths[:9]

# Display images
fig, axes = plt.subplots(3, 3, figsize=(10, 8))

for ax, image_path in zip(axes.flat, image_paths):
    image = Image.open(image_path)
    ax.imshow(image)
    ax.set_title(f"Class: {image_path.parent.name}")
    ax.axis("off")

plt.tight_layout()
plt.show()

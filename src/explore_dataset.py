
from pathlib import Path

dataset_path = Path(
    r"C:\TrafficSignProject\dataset\archive (1)"
)

train_path = dataset_path / "Train"
test_path = dataset_path / "Test"

print("Dataset exists:", dataset_path.exists())
print("Train folder exists:", train_path.exists())
print("Test folder exists:", test_path.exists())

if train_path.exists():
    classes = [p for p in train_path.iterdir() if p.is_dir()]
    print("Number of training classes:", len(classes))
    print("First 10 classes:", [p.name for p in classes[:10]])

    images = list(train_path.rglob("*.png"))
    print("PNG images found:", len(images))
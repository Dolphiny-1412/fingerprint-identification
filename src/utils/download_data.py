import kagglehub
from pathlib import Path
import shutil

# Download latest version
dataset_path = Path(kagglehub.dataset_download("shivamdapkekar/fingerprint-dataset-with-annotations"))
print("Path to dataset files:", dataset_path)

# Create new folder called "training data" in the current working directory (do not use fingerprintsClasess)
training_data_dir = Path.cwd() / "training data"
training_data_dir.mkdir(parents=True, exist_ok=True)
print("Created/using:", training_data_dir)

# Copy the downloaded dataset into "training data"
for item in dataset_path.iterdir():
    dest = training_data_dir / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)

print("Done.")


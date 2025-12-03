"""
Download Clown (Roboflow) + Shark (Open Images)
"""
import fiftyone as fo
import fiftyone.zoo as foz
from roboflow import Roboflow
from pathlib import Path

# 1. CLOWN da Roboflow
print("🤡 Downloading CLOWN from Roboflow...")

# Get API key: https://app.roboflow.com/settings/api
rf = Roboflow(api_key="pW9cJ2oAcSwkFRfNQImV")  # Sostituisci!

project = rf.workspace("phobicobjects").project("clown-q2kgo")
dataset = project.version(1).download("yolov5", location="data/raw/clown")

print("✅ Clown downloaded!")

# 2. SHARK da Open Images
print("\n🦈 Downloading SHARK from Open Images...")

shark_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Shark"],
    max_samples=400,
    dataset_name="sharks"
)
# --- DEBUGGING START ---
# Print the 10 most common labels found in the downloaded data
print("Top 10 labels found in ground_truth:")
from fiftyone import ViewField as F
# Count all values in the 'ground_truth.detections.label' field
counts = shark_dataset.count_values("ground_truth.detections.label")
print(counts)
# --- DEBUGGING END ---
shark_dataset.export(
    export_dir="data/raw/shark",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    #classes=["Shark"]
)

print("✅ Shark downloaded!")
print("\n📁 Next: python process_my_datasets.py")
import os
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import shutil

class DatasetValidator:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.issues = {
            "missing_label": [],
            "missing_image": [],
            "empty_label": [],
            "corrupt_image": []
        }

    def validate(self):
        """Checks for missing files, empty files, and corrupt images."""
        # reset stats
        for key in self.issues:
            self.issues[key] = []
            
        image_files = list(self.images_dir.glob("*"))
        label_files = list(self.labels_dir.glob("*"))
        
        # Maps purely filenames (no extension)
        img_stems = {f.stem: f for f in image_files}
        lbl_stems = {f.stem: f for f in label_files}

        # Check images
        for stem, img_path in img_stems.items():
            # 1. Check if label exists
            if stem not in lbl_stems:
                self.issues["missing_label"].append(str(img_path))
            
            # 2. Check if image opens (corruption check)
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    self.issues["corrupt_image"].append(str(img_path))
            except:
                self.issues["corrupt_image"].append(str(img_path))

        # Check labels
        for stem, lbl_path in lbl_stems.items():
            # 3. Check if image exists
            if stem not in img_stems:
                self.issues["missing_image"].append(str(lbl_path))
            
            # 4. Check if label file is empty
            if lbl_path.stat().st_size == 0:
                self.issues["empty_label"].append(str(lbl_path))

        return self.issues

    def print_report(self, stats):
        print(f"--- Validation Report for {self.images_dir.parent.name} ---")
        total_issues = sum(len(v) for v in stats.values())
        if total_issues == 0:
            print("✅ No issues found! Dataset is clean.")
        else:
            for issue_type, files in stats.items():
                if files:
                    print(f"❌ {issue_type}: {len(files)} files")

    def clean_dataset(self, output_dir):
        """Copies only valid image/label pairs to a new directory."""
        output_dir = Path(output_dir)
        out_imgs = output_dir / "images"
        out_lbls = output_dir / "labels"
        
        os.makedirs(out_imgs, exist_ok=True)
        os.makedirs(out_lbls, exist_ok=True)
        
        image_files = list(self.images_dir.glob("*"))
        
        copied_count = 0
        for img_path in image_files:
            stem = img_path.stem
            lbl_path = self.labels_dir / (stem + ".txt") # Assuming txt labels
            
            # Skip if any issue was recorded for this file
            has_issue = False
            for issue_list in self.issues.values():
                if str(img_path) in issue_list or str(lbl_path) in issue_list:
                    has_issue = True
                    break
            
            if not has_issue and lbl_path.exists():
                shutil.copy2(img_path, out_imgs / img_path.name)
                shutil.copy2(lbl_path, out_lbls / lbl_path.name)
                copied_count += 1
                
        print(f"✨ Cleaned dataset saved to {output_dir} ({copied_count} pairs copied)")

    def visualize_samples(self, num_samples=5):
        # Implementation for visualization if needed
        pass
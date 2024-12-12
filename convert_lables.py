import pandas as pd
import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split


class DatasetProcessor:
    def __init__(self, image_dir, label_dir, output_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

    def split_dataset(self, train_ratio=0.8, split_output_dir=None):
        if split_output_dir is None:
            split_output_dir = self.output_dir

        # Get all image and label files
        image_paths = glob(os.path.join(self.image_dir, "*.jpg"))
        label_paths = [
            os.path.join(self.label_dir, os.path.basename(p).replace(".jpg", ".txt"))
            for p in image_paths
        ]

        # Split the dataset
        train_images, val_images, train_labels, val_labels = train_test_split(
            image_paths, label_paths, train_size=train_ratio, random_state=42
        )

        # Define output directories
        dirs = {
            "train_images": os.path.join(split_output_dir, "images", "train"),
            "val_images": os.path.join(split_output_dir, "images", "val"),
            "train_labels": os.path.join(split_output_dir, "labels", "train"),
            "val_labels": os.path.join(split_output_dir, "labels", "val"),
        }

        # Create directories if they don't exist
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Copy files to the respective directories
        for img_path in train_images:
            shutil.copy(img_path, dirs["train_images"])
            lbl_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dirs["train_labels"])

        for img_path in val_images:
            shutil.copy(img_path, dirs["val_images"])
            lbl_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dirs["val_labels"])

        print("Dataset split completed.")

    def convert_labels(self, input_csv, label_output_dir=None):
        if label_output_dir is None:
            label_output_dir = self.label_dir

        # Read the CSV file
        df = pd.read_csv(input_csv)

        # Create output directory for label files if it doesn't exist
        os.makedirs(label_output_dir, exist_ok=True)

        # Process each unique image
        for image_id in df["image_id"].unique():
            # Get all bounding boxes for this image
            image_data = df[df["image_id"] == image_id]

            # Create a new txt file for this image
            with open(os.path.join(label_output_dir, f"{image_id}.txt"), "w") as f:
                for _, row in image_data.iterrows():
                    # Extract bbox string and convert to list of floats
                    bbox = eval(row["bbox"])  # Convert string "[x,y,w,h]" to list

                    # Original format: [x_min, y_min, width, height]
                    # YOLO format: [class x_center y_center width height] (normalized)

                    # Get image dimensions
                    img_width = row["width"]
                    img_height = row["height"]

                    # Convert coordinates
                    x_min = bbox[0]
                    y_min = bbox[1]
                    width = bbox[2]
                    height = bbox[3]

                    # Convert to YOLO format
                    # Convert bottom-left to top-left coordinate system
                    # y_min = img_height - (y_min + height)

                    # Calculate center points
                    x_center = (x_min + width / 2) / img_width
                    y_center = (y_min + height / 2) / img_height

                    # Normalize width and height
                    width = width / img_width
                    height = height / img_height

                    # Class is 0 as this is a single class problem (wheat)
                    class_id = 0

                    # Write to file (ensure values are within 0-1 range)
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

        print("Conversion completed!")


# Example usage
if __name__ == "__main__":
    input_csv = r"D:\code_local\Python\machine_learning_proj\wheat_dataset\train.csv"
    label_output_dir = r"D:\code_local\Python\machine_learning_proj\wheat_dataset\labels"
    image_dir = r"D:\code_local\Python\machine_learning_proj\wheat_dataset\train"
    split_output_dir = r"D:\code_local\Python\machine_learning_proj\dataset"

    processor = DatasetProcessor(image_dir, label_output_dir, split_output_dir)
    # processor.convert_labels(input_csv, label_output_dir)
    processor.split_dataset(split_output_dir=split_output_dir)  # Uncomment to split dataset

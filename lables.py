import os
import cv2
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    filename="label_generation.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Map class names to IDs (MUST match folder names)
class_map = {
    "hazardous": 0,
    "organic": 1,
    "recycleable": 2
}

# Supported image extensions
extensions = ["*.jpg", "*.jpeg", "*.png"]

def compute_bounding_box(image_path, debug_save_path=None):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning(f"Cannot read image {image_path}")
            return None

        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Default fallback: full image box
        x, y, w, h = 0, 0, img_w, img_h
        fallback = True

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area >= 500:
                x, y, w, h = cv2.boundingRect(largest)
                fallback = False
            else:
                logging.warning(f"Small object in {image_path}, using full image box.")
        else:
            logging.warning(f"No object in {image_path}, using full image box.")

        # Normalize to YOLO format
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h

        # Save debug image
        if debug_save_path:
            debug_img = img.copy()
            color = (0, 255, 0) if not fallback else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.imwrite(str(debug_save_path), debug_img)

        return x_center, y_center, width, height

    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None


def main(root_dir, debug=False):
    root = Path(root_dir)
    debug_dir = root / "debug_preview" if debug else None

    for split in ["train", "val", "test"]:
        for cls, cls_id in class_map.items():
            img_dir = root / split / cls
            lbl_dir = root / "labels" / split
            lbl_dir.mkdir(parents=True, exist_ok=True)

            img_paths = []
            for ext in extensions:
                img_paths.extend(img_dir.rglob(ext))  # <-- FIXED

            logging.info(f"[{split}] Class '{cls}' ({cls_id}): Found {len(img_paths)} images")
            print(f"[{split}] {cls}: found {len(img_paths)} images")

            for img_path in img_paths:
                stem = img_path.stem
                lbl_path = lbl_dir / f"{stem}.txt"

                debug_save_path = None
                if debug:
                    debug_save_path = debug_dir / split / cls / f"{stem}.jpg"
                    debug_save_path.parent.mkdir(parents=True, exist_ok=True)

                bbox = compute_bounding_box(img_path, debug_save_path)

                if bbox:
                    x_center, y_center, width, height = bbox
                    with open(lbl_path, "w") as f:
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    logging.warning(f"Skipped labeling {img_path.name} due to error")

    logging.info("✅ Label generation completed.")
    print("✅ Label generation completed. Check 'labels/' folder and debug_preview/ if enabled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Label Generator with Debug Mode")
    parser.add_argument("--root", type=str, default="C:/urmil/waste_dataset",
                        help="Root dataset directory")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug images showing bounding boxes")

    args = parser.parse_args()
    main(args.root, args.debug)

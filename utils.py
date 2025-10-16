import os
import csv
import datetime
import cv2
from collections import defaultdict

# Base directory for logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def _get_logfile(extension="log"):
    """Generate a daily log filename (detections_YYYY-MM-DD.ext)."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"detections_{today}.{extension}")


# ============================
# 1. Logging (TXT + CSV)
# ============================
def log_detection(label, yolo_conf, effnet_conf):
    """
    Append a detection to a daily .log (human-readable) and .csv (structured).
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Plain text log ---
    log_entry = f"[{timestamp}] {label} | YOLO={yolo_conf:.2f}, EffNet={effnet_conf:.2f}\n"
    with open(_get_logfile("log"), "a", encoding="utf-8") as f:
        f.write(log_entry)

    # --- CSV structured log ---
    csv_file = _get_logfile("csv")
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "label", "yolo_conf", "effnet_conf"])
        writer.writerow([timestamp, label, f"{yolo_conf:.4f}", f"{effnet_conf:.4f}"])


# ============================
# 2. Stats Tracker
# ============================
class StatsTracker:
    """
    Keeps track of detection counts per class and provides summary.
    """
    def __init__(self):
        self.counts = defaultdict(int)

    def update(self, label):
        """Increment count for a detected class."""
        self.counts[label] += 1

    def summary(self):
        """Return a dictionary summary of counts."""
        return dict(self.counts)

    def reset(self):
        """Reset all counts."""
        self.counts.clear()


# ============================
# 3. Overlay Helpers
# ============================
def draw_status(frame, fps, counts, position=(20, 40)):
    """
    Overlay FPS and class counts on the frame.
    """
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Class counts
    y_offset = position[1] + 30
    for label, count in counts.items():
        cv2.putText(frame, f"{label}: {count}", (position[0], y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25

    return frame


# ============================
# 4. Utility
# ============================
def get_timestamp():
    """
    Return a clean timestamp string (for filenames).
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

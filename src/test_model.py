from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n_ppe_best.pt"
IMAGE_DIR = PROJECT_ROOT / "assets" / "sample_images"

model = YOLO(str(MODEL_PATH))

# Select only five images
image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[:5]

results = model.predict(
    source=[str(path) for path in image_paths],
    conf=0.25,
    save=True
)

for result in results:
    print(f"\nImage: {Path(result.path).name}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"{model.names[class_id]}: {confidence:.2f}"
        )
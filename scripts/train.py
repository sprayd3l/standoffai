import os
import zipfile
import subprocess
import sys

def setup_and_train():
    # Extract dataset
    zip_path = "data/standoff.zip"
    extract_dir = "data/dataset"
    os.makedirs(extract_dir, exist_ok=True)

    print("[1/5] Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    # Fix paths in data.yaml (relative to yaml's directory)
    yaml_path = os.path.join(extract_dir, "data.yaml")
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()

    yaml_content = yaml_content.replace("../train/images", "train/images")
    yaml_content = yaml_content.replace("../valid/images", "valid/images")
    yaml_content = yaml_content.replace("../test/images", "test/images")

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("[2/5] Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])

    print("[3/5] Training YOLOv11n...")
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=20,
        project="runs",
        name="standoff2",
        exist_ok=True,
        verbose=True,
    )

    print("[4/5] Exporting to TFLite...")
    best_model = YOLO("runs/standoff2/weights/best.pt")
    best_model.export(format="tflite", imgsz=640)

    print("[5/5] Exporting to ONNX...")
    best_model.export(format="onnx", imgsz=640)

    print("DONE! Models in runs/standoff2/weights/")
    print("  - best.pt (PyTorch)")
    print("  - best.tflite (TFLite)")
    print("  - best.onnx (ONNX)")

    # Copy models to output
    os.makedirs("output", exist_ok=True)
    for f in ["best.pt", "best.tflite", "best.onnx"]:
        src = f"runs/standoff2/weights/{f}"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, f"output/{f}")

if __name__ == "__main__":
    setup_and_train()

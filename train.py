from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="model/data.yaml",
        epochs=50,
        batch=4,
        imgsz=800,
        device="cpu",
    )
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="model/data.yaml",
        epochs=120,
        batch=16,
        imgsz=800,
        device=0,
        patience = 25,
        save = True
    )                
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

image = cv2.imread("LotsEggs.jpg")

results = model(image)

annotated_image = results[0].plot()

cv2.imwrite("annotated_image.jpg", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
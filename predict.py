import cv2

from ultralytics import YOLO

model = YOLO(r".\best.pt")
results = model.predict(source = r"./images/06_0015.jpg", conf=0.25)

annotated_frame = results[0].plot()
output_destination = r".\predictions\prediction.jpg"

cv2.imwrite(output_destination, annotated_frame)
print(f"saved {output_destination}")

from ultralytics import YOLO

import cv2

model = YOLO("yolo26n.pt")

results = model.predict(
    source=("example.mp4"),
    stream=True,
    device=0, 
    classes=[2],
)

for result in results:
    annotated_frame = result.plot()
    cv2.imshow("车辆追踪",annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

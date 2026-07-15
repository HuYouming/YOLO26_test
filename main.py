import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv26 vehicle detection on mp4 videos")
    parser.add_argument("video_path", help="Input mp4 video path")
    parser.add_argument(
        "--output",
        default="output_video.mp4",
        help="Output video path (default: output_video.mp4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    model = YOLO("model/runs/detect/train/weights/best.pt")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    # 获取原始视频的属性，用于设置输出视频编码器
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, imgsz=1280, conf=0.20, device=0, classes=[0])
        annotated_frame = results[0].plot()

        # 显示
        cv2.imshow("YOLO 实时检测", annotated_frame)

        # 保存
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

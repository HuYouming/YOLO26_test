#!/usr/bin/env python3
"""
将 Cutie 生成的掩码图片转换为 YOLO 标签文件。

默认行为：
- 输入目录下所有常见图片格式都会被处理
- 单通道图按非零像素值区分目标，多通道图按非黑颜色区分目标
- 每个唯一像素值或颜色只生成一个外接矩形框
- 输出目录默认创建在当前项目目录下，名称为 labels_<输入目录名>

YOLO 标签格式：
class_id x_center y_center width height
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("参数必须是大于等于 0 的整数。")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Cutie 掩码图转换为 YOLO 数据集 labels 文本文件。"
    )
    parser.add_argument(
        "input_dir",
        help="包含掩码图的文件夹路径。",
    )
    parser.add_argument(
        "--class-id",
        type=non_negative_int,
        default=0,
        help="生成标签时使用的类别 ID，默认是 0。",
    )
    parser.add_argument(
        "--mask-value",
        type=non_negative_int,
        default=None,
        help="仅提取指定像素值对应的目标。适用于单通道、调色板或灰度+Alpha 掩码图。",
    )
    parser.add_argument(
        "--min-area",
        type=non_negative_int,
        default=1,
        help="忽略小于该像素面积的目标，默认是 1。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "输出目录名称或路径。未指定时，会在当前项目目录下创建 "
            "labels_<输入目录名>。"
        ),
    )
    return parser.parse_args()


def resolve_output_dir(input_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        return PROJECT_DIR / f"labels_{input_dir.name}"

    output_dir = Path(output_dir_arg).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    return output_dir.resolve()


def list_mask_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹: {input_dir}")

    mask_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not mask_files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"在 {input_dir} 中没有找到支持的掩码图文件，支持格式: {supported}"
        )

    stems: dict[str, Path] = {}
    for path in mask_files:
        if path.stem in stems:
            raise ValueError(
                f"发现同名文件会覆盖输出: {stems[path.stem].name} 和 {path.name}"
            )
        stems[path.stem] = path

    return mask_files


def load_mask_arrays(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, str, int, int]:
    try:
        with Image.open(image_path) as image:
            mask_array = np.array(image)
            color_array = np.array(image.convert("RGB")) if image.mode == "P" else None
            mode = image.mode
            width, height = image.size
    except UnidentifiedImageError as exc:
        raise ValueError(f"无法识别图片文件: {image_path}") from exc

    if mask_array.ndim not in {2, 3}:
        raise ValueError(f"不支持的图片维度: {image_path} -> ndim={mask_array.ndim}")

    return mask_array, color_array, mode, width, height


def box_from_mask(binary_mask: np.ndarray, min_area: int) -> tuple[int, int, int, int] | None:
    points_y, points_x = np.nonzero(binary_mask)
    area = points_y.size
    if area == 0 or area < min_area:
        return None

    return (
        int(points_x.min()),
        int(points_y.min()),
        int(points_x.max()),
        int(points_y.max()),
    )


def sort_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return sorted(boxes, key=lambda box: (box[1], box[0], box[3], box[2]))


def find_boxes_from_value_mask(
    value_mask: np.ndarray,
    min_area: int,
    mask_value: int | None,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []

    if mask_value is None:
        target_values = [int(value) for value in np.unique(value_mask) if int(value) != 0]
    else:
        target_values = [mask_value]

    for value in target_values:
        box = box_from_mask(value_mask == value, min_area=min_area)
        if box is not None:
            boxes.append(box)

    return sort_boxes(boxes)


def find_boxes_from_color_mask(
    color_mask: np.ndarray,
    min_area: int,
) -> list[tuple[int, int, int, int]]:
    if color_mask.ndim != 3 or color_mask.shape[2] < 3:
        raise ValueError(f"颜色掩码格式不支持: shape={color_mask.shape}")

    rgb = color_mask[:, :, :3].astype(np.uint32, copy=False)
    color_codes = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
    target_codes = [int(code) for code in np.unique(color_codes) if int(code) != 0]
    boxes: list[tuple[int, int, int, int]] = []

    for code in target_codes:
        box = box_from_mask(color_codes == code, min_area=min_area)
        if box is not None:
            boxes.append(box)

    return sort_boxes(boxes)


def find_bounding_boxes(
    mask_array: np.ndarray,
    color_array: np.ndarray | None,
    image_mode: str,
    min_area: int,
    mask_value: int | None,
) -> list[tuple[int, int, int, int]]:
    if mask_array.ndim == 2:
        if image_mode == "P" and mask_value is None:
            if color_array is None:
                raise ValueError("调色板图片缺少 RGB 颜色数据。")
            return find_boxes_from_color_mask(color_array, min_area=min_area)
        return find_boxes_from_value_mask(
            mask_array,
            min_area=min_area,
            mask_value=mask_value,
        )

    channel_count = mask_array.shape[2]
    if channel_count < 3:
        return find_boxes_from_value_mask(
            mask_array[:, :, 0],
            min_area=min_area,
            mask_value=mask_value,
        )

    if mask_value is not None:
        raise ValueError(
            f"{image_mode} 多通道图片不支持 --mask-value，请改用单通道/调色板掩码图。"
        )

    return find_boxes_from_color_mask(mask_array, min_area=min_area)


def to_yolo_line(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    class_id: int,
) -> str:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1
    x_center = (min_x + box_width / 2.0) / image_width
    y_center = (min_y + box_height / 2.0) / image_height

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{box_width / image_width:.6f} "
        f"{box_height / image_height:.6f}"
    )


def convert_masks_to_labels(
    input_dir: Path,
    output_dir: Path,
    class_id: int,
    mask_value: int | None,
    min_area: int,
) -> tuple[int, int]:
    mask_files = list_mask_files(input_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"输出路径不是文件夹: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total_boxes = 0

    for mask_path in mask_files:
        mask_array, color_array, image_mode, image_width, image_height = load_mask_arrays(
            mask_path
        )
        boxes = find_bounding_boxes(
            mask_array,
            color_array=color_array,
            image_mode=image_mode,
            min_area=min_area,
            mask_value=mask_value,
        )

        label_path = output_dir / f"{mask_path.stem}.txt"
        lines = [
            to_yolo_line(
                box,
                image_width=image_width,
                image_height=image_height,
                class_id=class_id,
            )
            for box in boxes
        ]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        processed_count += 1
        total_boxes += len(boxes)

    return processed_count, total_boxes


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = resolve_output_dir(input_dir, args.output_dir)

    try:
        processed_count, total_boxes = convert_masks_to_labels(
            input_dir=input_dir,
            output_dir=output_dir,
            class_id=args.class_id,
            mask_value=args.mask_value,
            min_area=args.min_area,
        )
    except Exception as exc:
        print(f"转换失败: {exc}", file=sys.stderr)
        return 1

    print(f"处理完成: {processed_count} 张掩码图")
    print(f"生成目录: {output_dir}")
    print(f"总目标框数: {total_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

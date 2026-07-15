import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cutie_mask_to_yolo import convert_masks_to_labels, find_bounding_boxes


class CutieMaskToYoloTests(unittest.TestCase):
    def test_rgb_mask_uses_one_box_per_color(self) -> None:
        mask = np.zeros((8, 8, 3), dtype=np.uint8)
        mask[1:3, 1:3] = [255, 0, 0]
        mask[4:6, 4:6] = [255, 0, 0]
        mask[5:7, 0:2] = [0, 255, 0]

        boxes = find_bounding_boxes(
            mask,
            color_array=None,
            image_mode="RGB",
            min_area=1,
            mask_value=None,
        )

        self.assertEqual(boxes, [(1, 1, 5, 5), (0, 5, 1, 6)])

    def test_grayscale_mask_uses_one_box_per_value(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 4
        mask[5:7, 5:7] = 4
        mask[2:5, 0:1] = 9

        boxes = find_bounding_boxes(
            mask,
            color_array=None,
            image_mode="L",
            min_area=1,
            mask_value=None,
        )

        self.assertEqual(boxes, [(1, 1, 6, 6), (0, 2, 0, 4)])

    def test_palette_mask_uses_visible_colors(self) -> None:
        image = Image.new("P", (8, 8), color=0)
        palette = [0, 0, 0] * 256
        palette[3:6] = [255, 0, 0]
        palette[6:9] = [0, 255, 0]
        image.putpalette(palette)

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[4:7, 4:6] = 1
        mask[0:2, 6:8] = 2
        image.putdata(mask.ravel().tolist())

        boxes = find_bounding_boxes(
            np.array(image),
            color_array=np.array(image.convert("RGB")),
            image_mode="P",
            min_area=1,
            mask_value=None,
        )

        self.assertEqual(boxes, [(6, 0, 7, 1), (1, 1, 5, 6)])

    def test_convert_masks_to_labels_writes_one_line_per_object_color(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "masks"
            output_dir = Path(temp_dir) / "labels"
            input_dir.mkdir()

            mask = np.zeros((10, 10, 3), dtype=np.uint8)
            mask[1:4, 1:4] = [255, 0, 0]
            mask[6:8, 6:9] = [255, 0, 0]
            mask[2:6, 7:9] = [0, 255, 0]
            Image.fromarray(mask, mode="RGB").save(input_dir / "sample.png")

            processed_count, total_boxes = convert_masks_to_labels(
                input_dir=input_dir,
                output_dir=output_dir,
                class_id=3,
                mask_value=None,
                min_area=1,
            )

            self.assertEqual(processed_count, 1)
            self.assertEqual(total_boxes, 2)
            self.assertEqual(
                (output_dir / "sample.txt").read_text(encoding="utf-8"),
                "3 0.500000 0.450000 0.800000 0.700000\n"
                "3 0.800000 0.400000 0.200000 0.400000\n",
            )


if __name__ == "__main__":
    unittest.main()

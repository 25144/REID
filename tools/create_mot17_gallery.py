import argparse
import os
from collections import defaultdict

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a ReID gallery folder from a MOT17 training sequence."
    )
    parser.add_argument(
        "--sequence_dir",
        required=True,
        type=str,
        help="MOT17 sequence directory, e.g. data/MOT17/images/train/MOT17-02-SDP.",
    )
    parser.add_argument(
        "--output_dir",
        default="gallery_mot17",
        type=str,
        help="Output gallery directory arranged as <identity_name>/*.jpg.",
    )
    parser.add_argument(
        "--max_images_per_id",
        default=5,
        type=int,
        help="Maximum reference crops saved for each identity.",
    )
    parser.add_argument(
        "--sample_stride",
        default=15,
        type=int,
        help="Only sample frames where frame_id %% sample_stride == 0.",
    )
    parser.add_argument("--min_box_area", default=1500.0, type=float)
    parser.add_argument("--min_visibility", default=0.0, type=float)
    parser.add_argument(
        "--require_pedestrian_class",
        action="store_true",
        help="Keep only MOT class id 1. Disabled by default for converted mirrors.",
    )
    return parser.parse_args()


def read_gt(gt_path):
    records_by_id = defaultdict(list)
    with open(gt_path, "r") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = [float(value) for value in parts[2:6]]
            class_id = int(float(parts[7]))
            visibility = float(parts[8])
            records_by_id[track_id].append(
                (frame_id, x, y, w, h, class_id, visibility)
            )
    return records_by_id


def crop_box(image, x, y, w, h):
    height, width = image.shape[:2]
    x1 = max(0, min(int(round(x)), width - 1))
    y1 = max(0, min(int(round(y)), height - 1))
    x2 = max(0, min(int(round(x + w)), width))
    y2 = max(0, min(int(round(y + h)), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def main():
    args = parse_args()
    img_dir = os.path.join(args.sequence_dir, "img1")
    gt_path = os.path.join(args.sequence_dir, "gt", "gt.txt")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Cannot find image directory: {img_dir}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Cannot find MOT17 ground truth file: {gt_path}")

    sequence_name = os.path.basename(os.path.normpath(args.sequence_dir))
    records_by_id = read_gt(gt_path)
    os.makedirs(args.output_dir, exist_ok=True)

    saved = 0
    for track_id, records in tqdm(sorted(records_by_id.items()), desc="Crop gallery"):
        identity_name = f"{sequence_name}_id_{track_id:04d}"
        identity_dir = os.path.join(args.output_dir, identity_name)
        os.makedirs(identity_dir, exist_ok=True)

        saved_for_id = 0
        for frame_id, x, y, w, h, class_id, visibility in records:
            if saved_for_id >= args.max_images_per_id:
                break
            if args.sample_stride > 1 and frame_id % args.sample_stride != 0:
                continue
            if w * h < args.min_box_area:
                continue
            if visibility < args.min_visibility:
                continue
            if args.require_pedestrian_class and class_id != 1:
                continue

            image_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
            image = cv2.imread(image_path)
            if image is None:
                continue
            crop = crop_box(image, x, y, w, h)
            if crop is None:
                continue

            output_path = os.path.join(identity_dir, f"{frame_id:06d}.jpg")
            cv2.imwrite(output_path, crop)
            saved_for_id += 1
            saved += 1

        if saved_for_id == 0:
            os.rmdir(identity_dir)

    print(f"Saved {saved} gallery crops to {args.output_dir}")


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from reid_utils import (  # noqa: E402
    build_reid_transform,
    crop_xyxy,
    extract_reid_features,
    load_config,
    load_reid_model,
    match_gallery,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 + ByteTrack + PCL-CLIP video person ReID demo."
    )
    parser.add_argument("--config_file", default="config/pcl-vit.yml", type=str)
    parser.add_argument(
        "--weight",
        default="logs/market-pclclip/ViT-B-16_60.pth",
        type=str,
        help="Path to the trained PCL-CLIP checkpoint.",
    )
    parser.add_argument(
        "--gallery_features",
        default="gallery_features.pt",
        type=str,
        help="Feature bank created by tools/build_gallery.py.",
    )
    parser.add_argument("--video", default=None, type=str, help="Input video path.")
    parser.add_argument(
        "--image_dir",
        default=None,
        type=str,
        help="Input image sequence directory, e.g. data/MOT17/train/MOT17-02-SDP/img1.",
    )
    parser.add_argument(
        "--output",
        default="outputs/video_reid.mp4",
        type=str,
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--detector",
        default="yolov8s.pt",
        type=str,
        help="YOLOv8 model name or path.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        type=str,
        help="Ultralytics tracker config. Use bytetrack.yaml for ByteTrack.",
    )
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--yolo_device", default=None, type=str)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument(
        "--fps",
        default=25.0,
        type=float,
        help="Output FPS for image sequence inputs such as MOT17 img1 folders.",
    )
    parser.add_argument("--sim_threshold", default=0.45, type=float)
    parser.add_argument(
        "--reid_interval",
        default=5,
        type=int,
        help="Extract ReID feature for each active track every N frames.",
    )
    parser.add_argument("--reid_batch_size", default=32, type=int)
    parser.add_argument("--camera_num", default=6, type=int)
    parser.add_argument("--view_num", default=1, type=int)
    parser.add_argument("--num_classes", default=None, type=int)
    parser.add_argument("--min_box_area", default=1000.0, type=float)
    parser.add_argument("--feature_momentum", default=0.8, type=float)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast.")
    parser.add_argument(
        "--no_gallery",
        action="store_true",
        help="Only draw YOLOv8/ByteTrack track ids without identity matching.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Optional config overrides, e.g. DATASETS.ROOT_DIR ./data.",
    )
    return parser.parse_args()


def load_yolo(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLOv8 detection. "
            "Install it with: pip install ultralytics"
        ) from exc
    return YOLO(model_path)


def load_gallery(path, disabled=False):
    if disabled:
        return [], torch.empty(0, 1280)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gallery feature file does not exist: {path}. "
            "Create it with tools/build_gallery.py, or pass --no_gallery."
        )
    gallery = torch.load(path, map_location="cpu")
    names = gallery["names"]
    features = F.normalize(gallery["features"].float(), dim=1)
    return names, features


def draw_label(frame, x1, y1, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_text = max(0, y1 - text_h - baseline - 4)
    cv2.rectangle(
        frame,
        (x1, y_text),
        (x1 + text_w + 4, y_text + text_h + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 2, y_text + text_h + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def color_for_track(track_id):
    palette = (
        (37, 255, 225),
        (255, 111, 89),
        (91, 192, 190),
        (255, 202, 58),
        (138, 201, 38),
        (106, 76, 147),
        (25, 130, 196),
    )
    return palette[int(track_id) % len(palette)]


def update_track_feature(track_features, track_id, feature, momentum):
    feature = F.normalize(feature.float().view(1, -1), dim=1).squeeze(0)
    if track_id in track_features:
        feature = F.normalize(
            momentum * track_features[track_id] + (1.0 - momentum) * feature,
            dim=0,
        )
    track_features[track_id] = feature.cpu()
    return track_features[track_id]


def list_image_frames(image_dir):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    frame_paths = []
    for pattern in patterns:
        frame_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    frame_paths = sorted(frame_paths)
    if not frame_paths:
        raise RuntimeError(f"No image frames were found in {image_dir}")
    return frame_paths


def prepare_frame_source(args):
    if bool(args.video) == bool(args.image_dir):
        raise ValueError("Pass exactly one input source: --video or --image_dir.")

    if args.video:
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or args.fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return capture, None, fps, width, height, frame_count

    frame_paths = list_image_frames(args.image_dir)
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise RuntimeError(f"Cannot read first image frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    return None, frame_paths, args.fps, width, height, len(frame_paths)


def main():
    args = parse_args()

    if args.no_gallery:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = None
        reid_model = None
    else:
        device = resolve_device(args.device)
        cfg = load_config(args.config_file, args.opts)
        transform = build_reid_transform(cfg)
        reid_model = load_reid_model(
            cfg,
            args.weight,
            device,
            num_classes=args.num_classes,
            camera_num=args.camera_num,
            view_num=args.view_num,
        )
    yolo_device = args.yolo_device or ("0" if device.type == "cuda" else "cpu")

    yolo = load_yolo(args.detector)
    gallery_names, gallery_features = load_gallery(
        args.gallery_features, disabled=args.no_gallery
    )

    capture, frame_paths, fps, width, height, frame_count = prepare_frame_source(args)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output}")

    track_features = {}
    track_labels = defaultdict(lambda: ("unknown", 0.0))
    frame_index = 0

    progress = tqdm(total=frame_count if frame_count > 0 else None, desc="Video ReID")
    try:
        while True:
            if capture is not None:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
            else:
                if frame_index >= len(frame_paths):
                    break
                frame_bgr = cv2.imread(frame_paths[frame_index])
                if frame_bgr is None:
                    raise RuntimeError(f"Cannot read image frame: {frame_paths[frame_index]}")

            results = yolo.track(
                frame_bgr,
                persist=True,
                tracker=args.tracker,
                classes=[0],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=yolo_device,
                verbose=False,
            )

            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    for box, track_id, score in zip(xyxy, track_ids, confs):
                        x1, y1, x2, y2 = box
                        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        if area < args.min_box_area:
                            continue
                        detections.append((track_id, box, float(score)))

            crops = []
            crop_track_ids = []
            if not args.no_gallery:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                for track_id, box, _ in detections:
                    if frame_index % max(1, args.reid_interval) != 0:
                        continue
                    crop = crop_xyxy(frame_rgb, box)
                    if crop is None:
                        continue
                    crops.append(crop)
                    crop_track_ids.append(track_id)

            if not args.no_gallery and crops:
                features = extract_reid_features(
                    reid_model,
                    crops,
                    transform,
                    device,
                    batch_size=args.reid_batch_size,
                    use_amp=args.amp,
                )
                for track_id, feature in zip(crop_track_ids, features):
                    smoothed_feature = update_track_feature(
                        track_features,
                        track_id,
                        feature,
                        args.feature_momentum,
                    )
                    name, similarity = match_gallery(
                        smoothed_feature, gallery_features, gallery_names
                    )
                    if args.no_gallery or similarity < args.sim_threshold:
                        track_labels[track_id] = ("unknown", similarity)
                    else:
                        track_labels[track_id] = (name, similarity)

            for track_id, box, det_score in detections:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                color = color_for_track(track_id)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                name, similarity = track_labels[track_id]
                if args.no_gallery:
                    label = f"track {track_id} det {det_score:.2f}"
                else:
                    label = f"id {track_id} {name} {similarity:.2f}"
                draw_label(frame_bgr, x1, y1, label, color)

            writer.write(frame_bgr)
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        if capture is not None:
            capture.release()
        writer.release()

    print(f"Saved annotated video to {args.output}")


if __name__ == "__main__":
    main()

import argparse
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from reid_utils import (
    build_reid_transform,
    extract_reid_features,
    iter_image_files,
    load_config,
    load_reid_model,
    read_rgb_image,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an identity gallery for video person re-identification."
    )
    parser.add_argument("--config_file", default="config/pcl-vit.yml", type=str)
    parser.add_argument(
        "--weight",
        default="logs/market-pclclip/ViT-B-16_60.pth",
        type=str,
        help="Path to the trained PCL-CLIP checkpoint.",
    )
    parser.add_argument(
        "--gallery_dir",
        default="gallery",
        type=str,
        help="Directory arranged as gallery/<identity_name>/*.jpg.",
    )
    parser.add_argument(
        "--output",
        default="gallery_features.pt",
        type=str,
        help="Output feature bank path.",
    )
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--camera_num", default=6, type=int)
    parser.add_argument("--view_num", default=1, type=int)
    parser.add_argument("--num_classes", default=None, type=int)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast.")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Optional config overrides, e.g. DATASETS.ROOT_DIR ./data.",
    )
    return parser.parse_args()


def collect_gallery_images(gallery_dir):
    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(
            f"Gallery directory does not exist: {gallery_dir}. "
            "Expected gallery/<identity_name>/*.jpg."
        )

    identity_to_paths = defaultdict(list)
    for image_path in iter_image_files(gallery_dir):
        rel_path = os.path.relpath(image_path, gallery_dir)
        parts = rel_path.split(os.sep)
        if len(parts) < 2:
            continue
        identity_to_paths[parts[0]].append(image_path)

    if not identity_to_paths:
        raise RuntimeError(
            f"No identity folders with images were found in {gallery_dir}. "
            "Expected gallery/<identity_name>/*.jpg."
        )

    return dict(sorted(identity_to_paths.items()))


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg = load_config(args.config_file, args.opts)
    transform = build_reid_transform(cfg)
    model = load_reid_model(
        cfg,
        args.weight,
        device,
        num_classes=args.num_classes,
        camera_num=args.camera_num,
        view_num=args.view_num,
    )

    identity_to_paths = collect_gallery_images(args.gallery_dir)
    names = []
    features = []
    source_images = {}

    for identity_name, image_paths in tqdm(
        identity_to_paths.items(), desc="Building gallery"
    ):
        images = [read_rgb_image(path) for path in image_paths]
        image_features = extract_reid_features(
            model,
            images,
            transform,
            device,
            batch_size=args.batch_size,
            use_amp=args.amp,
        )
        identity_feature = torch.nn.functional.normalize(
            image_features.mean(dim=0, keepdim=True), dim=1
        ).squeeze(0)
        names.append(identity_name)
        features.append(identity_feature)
        source_images[identity_name] = image_paths

    gallery = {
        "names": names,
        "features": torch.stack(features, dim=0),
        "source_images": source_images,
        "config_file": args.config_file,
        "weight": args.weight,
        "input_size": list(cfg.INPUT.SIZE_TEST),
        "pixel_mean": list(cfg.INPUT.PIXEL_MEAN),
        "pixel_std": list(cfg.INPUT.PIXEL_STD),
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(gallery, args.output)
    print(f"Saved {len(names)} identities to {args.output}")


if __name__ == "__main__":
    main()

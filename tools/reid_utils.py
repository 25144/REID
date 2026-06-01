import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import cfg as base_cfg  # noqa: E402
from pcl.model import make_model  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_file: str, opts: Optional[Sequence[str]] = None):
    cfg = base_cfg.clone()
    if config_file:
        cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(list(opts))
    cfg.freeze()
    return cfg


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(
            "This PCL-CLIP model currently requires CUDA because pcl/model.py "
            "moves the CLIP visual encoder to cuda during construction."
        )
    return resolved


def load_checkpoint(weight_path: str):
    checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {weight_path}")
    return checkpoint


def infer_num_classes(weight_path: str) -> int:
    checkpoint = load_checkpoint(weight_path)
    classifier_weight = checkpoint.get("classifier.weight")
    if classifier_weight is None:
        classifier_weight = checkpoint.get("module.classifier.weight")
    if classifier_weight is None:
        raise RuntimeError(
            "Cannot infer num_classes because classifier.weight was not found "
            f"in {weight_path}."
        )
    return int(classifier_weight.shape[0])


def build_reid_transform(cfg):
    return T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )


def load_reid_model(
    cfg,
    weight_path: str,
    device: torch.device,
    num_classes: Optional[int] = None,
    camera_num: int = 6,
    view_num: int = 1,
):
    if num_classes is None:
        num_classes = infer_num_classes(weight_path)

    model = make_model(cfg, num_classes, camera_num=camera_num, view_num=view_num)
    model.to(device)
    model.eval()

    checkpoint = load_checkpoint(weight_path)
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in checkpoint.items():
        clean_key = key.replace("module.", "")
        if clean_key.startswith("classifier."):
            skipped_keys.append(clean_key)
            continue
        if clean_key not in model_state:
            skipped_keys.append(clean_key)
            continue
        if model_state[clean_key].shape != value.shape:
            skipped_keys.append(clean_key)
            continue
        model_state[clean_key].copy_(value.to(model_state[clean_key].device))
        loaded_keys.append(clean_key)

    if not loaded_keys:
        raise RuntimeError(f"No model weights were loaded from {weight_path}")

    print(
        f"Loaded {len(loaded_keys)} keys from {weight_path}; "
        f"skipped {len(skipped_keys)} keys."
    )
    return model


def iter_image_files(root_dir: str) -> Iterable[str]:
    for current_root, _, files in os.walk(root_dir):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                yield os.path.join(current_root, filename)


def read_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def extract_reid_features(
    model,
    images: Sequence[Image.Image],
    transform,
    device: torch.device,
    batch_size: int = 64,
    use_amp: bool = False,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        batch = torch.stack([transform(image) for image in batch_images], dim=0)
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            feat = model(batch)
        feat = F.normalize(feat.float(), dim=1)
        features.append(feat.cpu())
    if not features:
        return torch.empty(0, 1280)
    return torch.cat(features, dim=0)


def crop_xyxy(frame_rgb, box: Sequence[float]) -> Optional[Image.Image]:
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return Image.fromarray(frame_rgb[y1:y2, x1:x2])


def match_gallery(
    feature: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_names: Sequence[str],
) -> Tuple[str, float]:
    if gallery_features.numel() == 0:
        return "unknown", 0.0
    feature = F.normalize(feature.float().view(1, -1), dim=1)
    gallery_features = F.normalize(gallery_features.float(), dim=1)
    similarities = torch.mm(feature.cpu(), gallery_features.cpu().t()).squeeze(0)
    score, index = torch.max(similarities, dim=0)
    return gallery_names[int(index)], float(score)

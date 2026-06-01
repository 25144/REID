# Video Example

Put local demo videos here when running `demo_video_reid.py`.

Example command:

```bash
python demo_video_reid.py \
  --config_file config/pcl-vit.yml \
  --weight logs/market-pclclip/ViT-B-16_60.pth \
  --gallery_features gallery_features.pt \
  --video examples/videos/input.mp4 \
  --output outputs/input_reid.mp4 \
  --detector yolov8s.pt \
  --tracker bytetrack.yaml
```

Large videos are ignored by Git and should not be committed.

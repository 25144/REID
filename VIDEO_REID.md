# YOLOv8 + ByteTrack + PCL-CLIP 视频行人识别追踪

本流程把视频行人识别拆成三步：

```text
YOLOv8 检测行人 -> ByteTrack 生成 track id -> PCL-CLIP 匹配身份图库
```

## 1. 安装额外依赖

先进入训练时使用的环境：

```bash
conda activate pclclip
pip install -r requirements-video.txt
```

`demo_video_reid.py` 使用 Ultralytics 的 YOLOv8 接口，并通过 `tracker=bytetrack.yaml` 启用 ByteTrack。

## 2. 准备身份图库

每个身份放一个子目录，目录名会作为视频里的身份标签：

```text
gallery/
  person_001/
    1.jpg
    2.jpg
  person_002/
    1.jpg
    2.jpg
```

图库图片建议使用清晰、完整的行人截图。每个人可以放多张，脚本会提取特征后求平均。

## 3. 构建图库特征

```bash
python tools/build_gallery.py \
  --config_file config/pcl-vit.yml \
  --weight logs/market-pclclip/ViT-B-16_60.pth \
  --gallery_dir gallery \
  --output gallery_features.pt
```

输出的 `gallery_features.pt` 会保存身份名、平均特征和来源图片路径。

## 4. 运行视频识别追踪

```bash
python demo_video_reid.py \
  --config_file config/pcl-vit.yml \
  --weight logs/market-pclclip/ViT-B-16_60.pth \
  --gallery_features gallery_features.pt \
  --video videos/input.mp4 \
  --output outputs/input_reid.mp4 \
  --detector yolov8s.pt \
  --tracker bytetrack.yaml \
  --sim_threshold 0.45
```

输出视频会显示：

- 行人检测框
- ByteTrack 的 `track_id`
- 匹配到的图库身份名
- ReID 相似度

## 5. 只验证检测和跟踪

如果还没有身份图库，可以先只看 YOLOv8 + ByteTrack 的跟踪结果：

```bash
python demo_video_reid.py \
  --config_file config/pcl-vit.yml \
  --weight logs/market-pclclip/ViT-B-16_60.pth \
  --video videos/input.mp4 \
  --output outputs/input_track.mp4 \
  --detector yolov8s.pt \
  --tracker bytetrack.yaml \
  --no_gallery
```

也可以直接用 Ultralytics 命令快速验证 ByteTrack：

```bash
yolo track model=yolov8s.pt source=videos/input.mp4 tracker=bytetrack.yaml
```

## 6. 使用 MOT17 做检测跟踪和 ReID

MOT17 通常是图像序列目录，不是单个视频文件。官方数据通常是 `train/` 和 `test/` 结构；如果使用 PaddleDetection 镜像，目录会是 `images/train/`、`images/test/` 和 `labels_with_ids/`：

```text
data/MOT17/
  images/
    train/
      MOT17-02-SDP/
        img1/
          000001.jpg
          000002.jpg
        gt/
          gt.txt
      MOT17-04-SDP/
        img1/
        gt/
    test/
    half/
  annotations/
  labels_with_ids/
```

只检测和跟踪某个 MOT17 序列：

```bash
python demo_video_reid.py \
  --image_dir data/MOT17/images/train/MOT17-02-SDP/img1 \
  --output outputs/MOT17-02-SDP_track.mp4 \
  --detector yolov8s.pt \
  --tracker bytetrack.yaml \
  --fps 30 \
  --no_gallery
```

### 6.1 从 MOT17 GT 自动生成身份图库

训练集序列带有 `gt/gt.txt`，可以用它自动裁剪每个真实行人 ID 的参考图库：

```bash
python tools/create_mot17_gallery.py \
  --sequence_dir data/MOT17/images/train/MOT17-02-SDP \
  --output_dir gallery_mot17/MOT17-02-SDP \
  --max_images_per_id 5 \
  --sample_stride 15 \
  --min_box_area 1500
```

输出目录格式会自动整理为：

```text
gallery_mot17/MOT17-02-SDP/
  MOT17-02-SDP_id_0001/
    000015.jpg
    000030.jpg
  MOT17-02-SDP_id_0002/
    000120.jpg
```

### 6.2 构建 MOT17 身份特征库

```bash
python tools/build_gallery.py \
  --config_file config/pcl-vit.yml \
  --weight logs/market-pclclip/ViT-B-16_60.pth \
  --gallery_dir gallery_mot17/MOT17-02-SDP \
  --output gallery_features_mot17_02.pt
```

### 6.3 在 MOT17 序列上输出 ReID 视频

有身份特征库后，可以在 MOT17 序列上同时做检测、跟踪和 ReID：

```bash
python demo_video_reid.py \
  --config_file config/pcl-vit.yml \
  --weight logs/market-pclclip/ViT-B-16_60.pth \
  --gallery_features gallery_features_mot17_02.pt \
  --image_dir data/MOT17/images/train/MOT17-02-SDP/img1 \
  --output outputs/MOT17-02-SDP_reid.mp4 \
  --detector yolov8s.pt \
  --tracker bytetrack.yaml \
  --fps 30 \
  --sim_threshold 0.45
```

MOT17 下载地址：[MOTChallenge MOT17](https://motchallenge.net/data/MOT17/)

如果官网下载较慢，也可以使用 PaddleDetection 文档里的镜像：

```bash
wget https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip -O data/MOT17.zip
unzip data/MOT17.zip -d data
```

## 7. 常用参数

- `--detector yolov8n.pt`：速度更快，检测精度较低。
- `--detector yolov8s.pt`：默认推荐，速度和精度比较均衡。
- `--detector yolov8m.pt`：检测更稳，但更吃显存。
- `--sim_threshold 0.45`：身份匹配阈值。误识别多就调高，识别不到就调低。
- `--reid_interval 5`：每条轨迹每隔多少帧提一次 ReID 特征。调大更快，调小更稳。
- `--min_box_area 1000`：过滤太小的行人框。
- `--amp`：用半精度推理，通常更快。

## 8. 注意事项

- `logs/market-pclclip/ViT-B-16_60.pth` 是 Market1501 行人 ReID 权重，只适合行人外观再识别。
- ReID 不是人脸识别，主要依赖衣服、背包、体型等外观信息；换衣服后通常无法稳定识别。
- 当前 PCL-CLIP 模型构建代码会把 CLIP 视觉编码器放到 CUDA，因此这套脚本需要 NVIDIA GPU。
- YOLOv8 权重负责检测，PCL-CLIP 权重负责身份特征匹配，两者不是同一个模型。
- `--no_gallery` 模式只运行 YOLOv8 + ByteTrack，不会加载 PCL-CLIP ReID 模型，适合先在 MOT17 上验证检测和跟踪。

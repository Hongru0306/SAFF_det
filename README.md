# SAFF: A Spatially-Aware Fusion Framework for Effective and Efficient Aerial Object Detection

This repository contains the official implementation of **SAFF**, a multimodal (RGB + IR) aerial object detection framework built on top of [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).

---

## Environment

This project is based on the [Ultralytics](https://github.com/ultralytics/ultralytics) framework. Please follow their installation guide to set up the environment, then install any additional dependencies:



---

## Project Structure

```
SAFF/
├── data/               # Dataset config files (.yaml)
├── configs/  # Model architecture configs
├── weights/            # Pretrained weights and training checkpoints
│   ├── saff-n/
│   └── saff-m/
├── train.py            # Training script
└── val.py              # Validation/testing script
```

---

## Training

Edit the model config path and dataset config path in `train.py`, then run:

```bash
python train.py
```

Key arguments (configured inside `train.py`):

| Argument | Description |
|---|---|
| `data` | Path to dataset `.yaml` config |
| `imgsz` | Input image size (default: 640) |
| `epochs` | Number of training epochs |
| `batch` | Batch size |
| `device` | GPU device id(s) |
| `project` | Output directory for runs |

---

## Validation

Edit the model weight path and dataset config path in `val.py`, then run:

```bash
python val.py
```

---

## Model Configs

Model architecture configs are located in `./multimodal_models/`. The backbone accepts 6-channel input (RGB + IR concatenated) and supports oriented bounding box (OBB) detection.

---

## Weights

Pretrained weights and training logs for two model scales are provided in `./weights/`:

| Model | Directory |
|---|---|
| SAFF-N (nano) | `./weights/saff-n/` |
| SAFF-M (medium) | `./weights/saff-m/` |

---

## Citation
```
@ARTICLE{11436060,
  author={Xiao, Hongru and Zhuang, Jiankun and Yang, Bin and Hu, Jinming and Zhu, Junze and Lian, Zhen and Zhao, Sijie and Zhou, Yanmin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SAFF: A Spatially-Aware Fusion Framework for Effective and Efficient Aerial Object Detection}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Accuracy;Semantics;Object detection;Feature extraction;Computational efficiency;Location awareness;Detectors;Computational modeling;Costs;Lighting;Aerial object detection;Multimodal fusion;Infrared-RGB Detection;Deep learning},
  doi={10.1109/TGRS.2026.3674467}}
```

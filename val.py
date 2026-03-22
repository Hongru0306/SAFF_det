from ultralytics import YOLO, RTDETR
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == '__main__':

    
    model = YOLO("/share/workspace/lixiang/XiaoHongru/project/paper/IRdetect/AAAI_Els/SAFF/runs/tgrs/train/weights/best.pt")
    model.val(data="/share/workspace/lixiang/XiaoHongru/project/paper/IRdetect/AAAI_Els/SAFF/data/VEDAI_rgb.yaml",batch=16,
              split='test')



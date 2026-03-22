from ultralytics import YOLO, RTDETR
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == '__main__':
    # 训练  
    model = YOLO('/share/workspace/lixiang/XiaoHongru/project/paper/IRdetect/AAAI_Els/SAFF/multimodal_models/tgrs_obb_11m_15.yaml')

    
    model.train(data='/share/workspace/lixiang/XiaoHongru/project/paper/IRdetect/v11multi_det_obb/data/VEDAI_rgb.yaml',
                cache=False,
                imgsz=640,
                epochs=2,
                batch=16,
                workers=8, 
                device='0',
                patience=400, # set 0 to close earlystop.
                pretrained=False, # 使用
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/tgrs',
                )



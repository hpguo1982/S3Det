import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                )
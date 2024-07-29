import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.val(data='',
              split='',
              imgsz=640,
              batch=16,
              )
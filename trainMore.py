import os 

from ultralytics import YOLOv10 

config_path = '/home/xavier/Robolab/config.yaml'

model = YOLOv10('/home/xavier/Robolab/runs/detect/train5/weights/last.pt')

model.train(data=config_path, epochs = 150, batch = 32)
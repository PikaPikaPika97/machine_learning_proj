from ultralytics import YOLO
from dataset import yolo_predictions_to_csv

# Training configuration
training_configuration = {
    "data": ".\dataset.yaml",
    "epochs": 2,
    "batch": 8,
    "imgsz": 1024,
    "name" : "train_test",
    "exist_ok": True,
    "optimizer": "Adam",
    "single_cls": False,
    "cos_lr": True,
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 3,
}

augmentation_configuration = {
    "hsv_h":0.015,
    "hsv_s":0.7,
    "hsv_v":0.4,
    "degrees":40.0,
    "translate":0.1,
    "scale":0.5,
    "shear":20.0,
    "perspective":0.0005,
    "flipud":0.5,
    "fliplr":0.5,
    "mosaic":1.0,
    "mixup":1.0,
}

training_configuration.update(augmentation_configuration)

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
train_results = model.train(**training_configuration)



# model_path = "/root/code/machine_learning_proj/runs/detect/train_200_epochs/weights/best.pt"
# predicting_configuration = {
#     "source": "/root/autodl-tmp/wheat_dataset/test/*.jpg",
#     "conf": 0.25,
#     "iou": 0.7,
#     "imgsz": 1024,
#     "max_det": 300,
#     "augment": False,
#     "name": None,
#     "show": False,
#     "save": True,
#     "show_labels": False,
# }

# model = YOLO(model_path)
# results = model.predict(**predicting_configuration)
# yolo_predictions_to_csv(results, "submission.csv")
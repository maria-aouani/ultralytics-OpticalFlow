from ultralytics import YOLO

# Load the modified model
model = YOLO("yolov8-pose_my_model_conf.yaml")

# Train the model
results = model.train(
    data="kazakh_sign_language.yaml",
    epochs=3,
    batch=16,
    imgsz=640,
    name='custom_training_run'
)

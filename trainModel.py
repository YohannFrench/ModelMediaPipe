import os
import json
import tensorflow as tf
from mediapipe_model_maker import object_detector
from mediapipe_model_maker import quantization

train_dataset_path = "dataset_persons/train"
validation_dataset_path = "dataset_persons/valid"
test_dataset_path = "dataset_persons/test"

with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
    labels_json = json.load(f)
for category_item in labels_json["categories"]:
    print(f"{category_item['id']}: {category_item['name']}")

train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="../cache/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="../cache/valid")
test_data = object_detector.Dataset.from_coco_folder(test_dataset_path, cache_dir="../cache/test")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)
print("test_data_size: ", test_data.size)

spec = object_detector.SupportedModels.MOBILENET_V2
hparams = object_detector.HParams(epochs=2, batch_size=32, export_dir='exported_model')
options = object_detector.ObjectDetectorOptions(supported_model=spec, hparams=hparams)

model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)

loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")

# Application of post-training quantization to reduce model size and improve inference speed for smaller devices
quantization_config = quantization.QuantizationConfig.for_int8(representative_data=test_data)

model.export_model(model_name="default_model.tflite")
model.export_model(model_name='model_int8.tflite', quantization_config=quantization_config)
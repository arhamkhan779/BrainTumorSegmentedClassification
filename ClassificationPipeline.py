import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from SegmentationPipeline import predict_segmentation_mask
model_path  = Path("classification_model.keras")

def load_model(path:str):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        raise e
    
model = load_model(model_path)
def saliency_map_prediction(model, image_path, target_size=(224, 224)):
    # Loading image from directory
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(image, target_size)
    img_normalized = img_resized / 255.0
    
    # Add channel dimension for grayscale: (224, 224) -> (224, 224, 1)
    img_normalized = np.expand_dims(img_normalized, axis=-1)
    
    # Add batch dimension: (224, 224, 1) -> (1, 224, 224, 1)
    img_input = np.expand_dims(img_normalized, axis=0)
    
    class_names = ['pituitary', 'notumor', 'meningioma', 'glioma']

    # Perform prediction using model
    prediction = model.predict(img_input)
    pred_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[pred_class_idx]
    
    # Compute saliency map
    img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        top_class = preds[0][pred_class_idx]  # Use the predicted class score
    
    grads = tape.gradient(top_class, img_tensor)
    
    # For grayscale images, squeeze the batch and channel dimensions
    # Shape: (1, 224, 224, 1) -> (224, 224)
    saliency = tf.abs(grads[0, :, :, 0]).numpy()
    return predicted_label,saliency,img_resized


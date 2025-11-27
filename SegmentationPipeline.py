import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

model_path  = Path("segmentation_unet_model.keras")

def load_model(path:str):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        raise e
    
model = load_model(model_path)




def predict_segmentation_mask(model, image_path, target_size=(256, 256)):
    """
    Predict segmentation mask for U-Net and overlay purple mask.
    """

    # 1. Load & preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig_resized = cv2.resize(image, target_size)
    img_norm = orig_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=(0, -1))

    # 2. Predict mask
    pred_mask = model.predict(img_input)[0]  # shape: (H, W, 1)

    # 3. Convert to binary mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # 4. Prepare overlay in purple color (BGR: 128, 0, 128)
    purple_layer = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    purple_layer[..., 0] = 128  # Blue
    purple_layer[..., 1] = 0    # Green
    purple_layer[..., 2] = 128  # Red

    # 5. Apply mask to purple layer
    purple_mask = (binary_mask * purple_layer).astype(np.uint8)

    # 6. Convert original grayscale to 3-channel
    gray_3ch = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)

    # 7. Blend images
    overlay = cv2.addWeighted(gray_3ch, 0.7, purple_mask, 0.3, 0)
    tumor_area_pixels = np.sum(binary_mask)  # number of white pixels
    # Optional: equivalent circular diameter in pixels
    tumor_diameter_pixels = 2 * np.sqrt(tumor_area_pixels / np.pi)

    return orig_resized, (binary_mask * 255).squeeze().astype(np.uint8), overlay, tumor_diameter_pixels




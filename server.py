from ClassificationPipeline import saliency_map_prediction
from SegmentationPipeline import predict_segmentation_mask
from ClassificationPipeline import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # ADD THIS
import shutil
import os
import uuid
import base64
import cv2
import numpy as np

def encode_image(image, is_saliency=False):
    """Encode image to base64 with compression"""
    
    # Special handling for saliency maps
    if is_saliency:
        # Normalize saliency map to 0-255 range
        if image.dtype != np.uint8:
            # Normalize to 0-1 first
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            # Scale to 0-255
            image = (image * 255).astype(np.uint8)
        
        # Apply colormap for better visualization (optional but recommended)
        if len(image.shape) == 2:  # If grayscale
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
    # Resize if image is too large
    max_size = 800
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    # Encode with quality setting
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

try:
    seg_model = load_model("segmentation_unet_model.keras")
    cls_model = load_model("classification_model.keras")
except Exception as e:
    raise e

def segmented_classify(image_path: str):
    _, binary_mask, mask_overlay, tumor_size = predict_segmentation_mask(
        model=seg_model, image_path=image_path
    )
    predicted_label, saliency_map, original_image = saliency_map_prediction(
        model=cls_model, image_path=image_path
    )
    return original_image, mask_overlay, saliency_map, round(tumor_size), predicted_label


app = FastAPI()

# ADD CORS MIDDLEWARE - CRITICAL!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}.jpg"
    temp_filepath = os.path.join(temp_dir, temp_filename)

    try:
        # Save uploaded file
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        print("Processing image...")
        
        # Get predictions
        original_image, mask_overlay, saliency_map, tumor_size, predicted_label = segmented_classify(temp_filepath)

        print("Encoding images...")
        print(f"Saliency map - Shape: {saliency_map.shape}, dtype: {saliency_map.dtype}, min: {saliency_map.min()}, max: {saliency_map.max()}")
        
        # Encode images to base64
        original_b64 = encode_image(original_image, is_saliency=False)
        mask_b64 = encode_image(mask_overlay, is_saliency=False)
        saliency_b64 = encode_image(saliency_map, is_saliency=True)  # Mark as saliency

        print("Sending response...")
        
        # Return response
        return JSONResponse(
            content={
                "predicted_label": str(predicted_label),
                "tumor_size": int(tumor_size),
                "original_image": original_b64,
                "mask_overlay": mask_b64,
                "saliency_map": saliency_b64
            },
            status_code=200
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
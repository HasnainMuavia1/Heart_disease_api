from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from tempfile import NamedTemporaryFile
import uuid
from pathlib import Path

# Import the prediction function from predict_ecg.py
from predict_ecg import predict_path

app = FastAPI()


@app.post("/predict/")
async def predict_ecg(file: UploadFile = File(...)):
    """
    Endpoint to predict ECG image.
    Accepts an image file and returns prediction results as JSON.
    """
    try:
        # Create a temporary file to store the uploaded image
        temp_file = NamedTemporaryFile(delete=False, suffix='.png')
        
        # Copy the uploaded file to the temporary file
        with temp_file as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get the path of the temporary file
        temp_path = Path(temp_file.name)
        
        # Use the predict_path function from predict_ecg.py
        _, pred_class, _, confidence = predict_path(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Return the prediction as JSON
        return {
            "prediction": pred_class,
            "confidence": float(confidence)
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

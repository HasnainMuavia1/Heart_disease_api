# Heart Disease FastAPI

A FastAPI application for heart disease prediction using ECG images with a VGG16-based deep learning model.

## Project Structure

```
Heart_disease/
├── main.py                     # FastAPI application entry point
├── predict_ecg.py              # ECG prediction module with model loading and inference
├── my_ecg_model_vgg16.pth      # Pre-trained VGG16 model for ECG classification
├── requirements.txt            # Project dependencies
└── test/                       # Directory containing test ECG images
```

## Features

- REST API for heart disease prediction from ECG images
- Uses a fine-tuned VGG16 model for classification
- Returns prediction class and confidence score

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HasnainMuavia1/Heart_disease_api.git
   cd Heart_disease_api
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
   # Additionally, install FastAPI and Uvicorn
   pip install fastapi uvicorn python-multipart
   ```

## Running the API

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. The API will be available at `http://127.0.0.1:8000`

## API Endpoints

### POST /predict/

Accepts an ECG image file and returns a prediction.

**Request:**
- Method: POST
- Endpoint: `/predict/`
- Body: Form data with a file field named `file` containing the ECG image

**Response:**
```json
{
  "prediction": "class_name",
  "confidence": 0.9876
}
```

## Interactive API Documentation

FastAPI provides automatic interactive API documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model Information

The prediction model is a fine-tuned VGG16 neural network trained on ECG images to classify heart conditions. The model file (`my_ecg_model_vgg16.pth`) contains:

- Model state dictionary
- Class-to-index mapping for classification

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful prediction
- 500: Server error with error message

## License

[MIT License](LICENSE)

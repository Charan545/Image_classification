from django.shortcuts import render
from tensorflow.keras.models import load_model
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from PIL import Image
from io import BytesIO
import os

# Path to the model
model_path = os.path.join(settings.BASE_DIR, 'classification_app/models/mobilenet_model.h5')
model = load_model(model_path)

# Class names corresponding to the model's predictions
CLASS_NAMES = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
    95: "Bird",
    948: "Apple",
    291: "Lion",
    282: "Tiger"
    # Add more if needed based on your model's class indices
}

def home(request):
    # Render the home.html template
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        # Check if an image is uploaded
        image_data = request.FILES.get('image')
        if not image_data:
            return JsonResponse({'error': 'No image uploaded'}, status=400)

        try:
            # Process the uploaded image
            img = Image.open(BytesIO(image_data.read()))

            # Check if the image has an alpha channel and remove it if present
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img = img.resize((224, 224))  # Resize to model's expected input size
            img_array = np.array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the class of the image
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))  # Convert np.int64 to Python int
            predicted_label = CLASS_NAMES.get(predicted_class, "Unknown")  # Map class to label
            confidence_score = float(np.max(prediction))  # Get the confidence score (probability)

            # Return the prediction and confidence as a JSON response
            return JsonResponse({
                'predicted_class': predicted_class,
                'label': predicted_label,
                'confidence': round(confidence_score, 2)  # Rounded to two decimal places
            })
        except Exception as e:
            # Handle any errors during prediction
            return JsonResponse({'error': str(e)}, status=500)
    
    # Render the upload image page if not a POST request
    return render(request, 'upload_image.html')

def serve_media(request, path):
    # Serve media files if needed
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return HttpResponse(file.read(), content_type="application/octet-stream")
    else:
        return HttpResponse("File not found", status=404)

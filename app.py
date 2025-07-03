import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Load the trained EfficientNetV2B2 model
model = tf.keras.models.load_model("Effiicientnetv2b2.keras")

# Define your class names in the order used during model training
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to classify the input image
def classify_image(img):  
    # Resize image to 224x224 (standard for EfficientNetV2B2)
    img = img.resize((124, 124))  
    
    # Convert to NumPy array and preprocess using EfficientNetV2 preprocessing
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)  
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  
    
    # Predict
    prediction = model.predict(img_array)  
    
    # Get predicted class and confidence
    predicted_class_index = np.argmax(prediction)  
    predicted_class_name = class_names[predicted_class_index]  
    confidence = prediction[0][predicted_class_index]  
    
    return f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})"

# Gradio Interface
iface = gr.Interface(  
    fn=classify_image,  
    inputs=gr.Image(type="pil"),  
    outputs="text",
    title="Garbage Classification - EfficientNetV2B2",
    description="Upload an image of a garbage item (metal, paper, plastic, etc.) to classify."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

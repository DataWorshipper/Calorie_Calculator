from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import requests
import json
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# Initialize ViT model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# API Key for nutritional information
api_key = "uTcpXk8HrKpe31UfvZs8Mg==8ofEmBLJjwJ9qCde"

def food_identification(image_path):
    """Identifies the food from an image using ViT model."""
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    food_name = predicted_label.split(',')[0]
    return food_name

def calorie_calculator(food_name):
    """Fetches nutritional information using an API."""
    api_url = f'https://api.api-ninjas.com/v1/nutrition?query={food_name}'
    response = requests.get(api_url, headers={'X-Api-Key': api_key})
    if response.status_code == requests.codes.ok:
        nutrition_info = json.loads(response.text)
    else:
        nutrition_info = {"error": response.status_code, "message": response.text}
    return nutrition_info

def format_nutrition_info(nutrition_info):
    """Formats nutritional information into HTML table."""
    if "error" in nutrition_info:
        return f"<p>Error fetching nutritional information: {nutrition_info['message']}</p>"
    
    if not nutrition_info:
        return "<p>No nutritional information found.</p>"
    
    table = f'''<table>
  <tr><td colspan="4" style="text-align: center;"><b>Food Name: {nutrition_info[0]['name']}</b></td></tr>
  <tr><td>Total Fat (g)</td><td>{nutrition_info[0]['fat_total_g']}</td><td>Saturated Fat (g)</td><td>{nutrition_info[0]['fat_saturated_g']}</td></tr>
  <tr><td>Sodium (mg)</td><td>{nutrition_info[0]['sodium_mg']}</td></tr>
  <tr><td>Potassium (mg)</td><td>{nutrition_info[0]['potassium_mg']}</td><td>Cholesterol (mg)</td><td>{nutrition_info[0]['cholesterol_mg']}</td></tr>
  <tr><td>Total Carbohydrates (g)</td><td>{nutrition_info[0]['carbohydrates_total_g']}</td><td>Fiber (g)</td><td>{nutrition_info[0]['fiber_g']}</td></tr>
  <tr><td>Sugar (g)</td><td>{nutrition_info[0]['sugar_g']}</td><td></td><td></td></tr>
</table>
'''
    
    return table

def gradio_interface(image):
    """Gradio interface function to process image and display nutritional information."""
    food_name = food_identification(image)
    nutrition_info = calorie_calculator(food_name)
    tabular_nutrition_info = format_nutrition_info(nutrition_info)
    return tabular_nutrition_info

# Define Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="filepath", label="Upload an image of food"),
    outputs="html",  # Specify output type directly as "html"
    title="Food Identification and Nutritional Value",
    description="Upload an image of food to get nutritional information",
    allow_flagging=False  # Disable flagging for simplicity
)

if __name__ == "__main__":
    iface.launch()

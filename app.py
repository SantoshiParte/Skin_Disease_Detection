from flask import Flask, render_template, request, jsonify
import torch
from torchvision.transforms import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import os

app = Flask(__name__)

#load class names
class_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
                'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 
                'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 
                'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 
                'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 
                'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis',
                  'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites',
                    'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 
                    'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives']

#Causes and Precautions dictionary

info_dict = {
    'Acne and Rosacea Photos': {
        'causes': 'Hormonal changes, bacteria, genetics, stress, diet, and certain medications can contribute to acne and rosacea.',
        'precautions': 'Maintain a gentle skincare routine, avoid harsh products, manage stress, and consult a dermatologist for appropriate treatment.'
    },
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': {
        'causes': 'Prolonged sun exposure, fair skin, age, and a history of sunburns can increase the risk of actinic keratosis and skin cancer.',
        'precautions': 'Use sunscreen, wear protective clothing, avoid tanning beds, and regularly check your skin for any changes or new growths.'
    },
    'Atopic Dermatitis Photos': {
        'causes': 'Genetics, environmental factors, and immune system dysfunction can contribute to atopic dermatitis.',
        'precautions': 'Maintain a consistent skincare routine, avoid known triggers, and consult a dermatologist for appropriate treatment.'
    },
    'Bullous Disease Photos': {
        'causes': 'Autoimmune disorders, genetic factors, and certain medications can lead to bullous diseases.',
        'precautions': 'Seek medical attention for proper diagnosis and treatment, avoid trauma to the skin, and follow prescribed medications and care routines.'  
    },
    'Cellulitis Impetigo and other Bacterial Infections': {
        'causes': 'Bacterial infections, often caused by Staphylococcus or Streptococcus bacteria, can lead to cellulitis and impetigo.',
        'precautions': 'Keep wounds clean and covered, practice good hygiene, and seek medical attention for proper diagnosis and treatment.'
    },
    'Eczema Photos': {
        'causes': 'Genetics, environmental factors, and immune system dysfunction can contribute to eczema.',
        'precautions': 'Maintain a consistent skincare routine, avoid known triggers, and consult a dermatologist for appropriate treatment.'
    },
    'Seborrheic Dermatitis Photos': {
        'causes': 'Genetics, hormonal changes, and the presence of a yeast-like fungus can contribute to seborrheic dermatitis.',
        'precautions': 'Use gentle, fragrance-free products, manage stress, and consult a dermatologist for appropriate treatment.'
    },
    'Exanthems and Drug Eruptions': {
        'causes': 'Viral infections, medications, and allergic reactions can lead to exanthems and drug eruptions.',
        'precautions': 'Identify and avoid triggers, follow prescribed medications, and seek medical attention for proper diagnosis and treatment.'         
    },
    'Hair Loss Photos Alopecia and other Hair Diseases': {
        'causes': 'Genetics, hormonal changes, autoimmune disorders, and certain medications can contribute to hair loss and alopecia.',
        'precautions': 'Maintain a healthy diet, manage stress, avoid harsh hair treatments,    and consult a dermatologist for appropriate treatment.'     
    },
    'Herpes HPV and other STDs Photos': {
        'causes': 'Sexually transmitted infections, including herpes and HPV, can lead to various skin conditions.',
        'precautions': 'Practice safe sex, get vaccinated for HPV, and seek medical attention for proper diagnosis and treatment of any suspected infections.'  
    },
}

#load Feature Extracter
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224in21k")

#Load Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224in21k",
    num_labels=len(class_names)
)
model.load_state_dict(torch.load("vit_model_pth", map_location=torch.device("cpu")))
model.eval()

#image Preprocessing
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(image_path):
    image = process_image(image_path)
    with torch.no_grad():
        outputs = model(image).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100 # Convert to percentage

    cause = info_dict.get(predicted_label, {}).get("cause","Unknown")
    precaution = info_dict.get(predicted_label, {}).get("predictions", "No specific precautions available.")

    return predicted_label, confidence_score, cause, precaution

#Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename =="":
            return render_template("index.html", error="No file selected")

        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        predicted_label, confidence_score, cause, precaution = predict_image(file_path)

        return render_template(
            "index.html",uploaded_image =file_path,
            predicted_label = predicted_label, confidence_score = confidence_score,
            cause = cause, precaution=precaution
        )

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from google.oauth2 import id_token
from google.auth.transport import requests
import json
import base64
from torchvision import models, transforms
from PIL import Image
import io
import torch
import torch.nn as nn
import bcrypt

app = Flask(__name__)
CORS(app)

# MongoDB setup
client = MongoClient('mongodb+srv://emrannooremon70:emon1234@nutritionapp.lecyp.mongodb.net/nutritionApp?retryWrites=true&w=majority')
db = client['nutritionApp']
users_collection = db['users']

# Load calorie data from calorie_map.json
with open("database/calorie_map.json", 'r') as f:
    calorie_map = json.load(f)

# Define the model architecture
def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model

# Load model weights
def load_model():
    model = create_model()
    state_dict = torch.load("database/model/enhanced_trained_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Initialize model
model = load_model()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Helper function to format food names
def format_food_name(food_name):
    return food_name.replace('_', ' ').title()

# Function to predict image class
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    predicted_class = predicted.item()
    return predicted_class

# Helper function to verify Google ID token
def verify_google_token(token):
    try:
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), "102622519531-5k1uvtvrh4bibs1c13uel8fq7fe2k09m.apps.googleusercontent.com")
        return idinfo
    except ValueError:
        return None

# Auth endpoint that handles both login and signup
@app.route('/auth', methods=['POST'])
def auth():
    data = request.get_json()
    google_token = data.get('google_token')

    # Verify Google token
    idinfo = verify_google_token(google_token)
    if idinfo is None:
        return jsonify({"error": "Invalid Google token"}), 401

    email = idinfo.get('email')
    name = idinfo.get('name')
    google_id = idinfo.get('sub')

    # Check if user already exists
    user = users_collection.find_one({"google_id": google_id})
    if not user:
        user_data = {
            "name": name,
            "email": email,
            "google_id": google_id
        }
        users_collection.insert_one(user_data)
        return jsonify({"message": "User signed up with Google", "user_id": str(user_data['_id'])}), 201

    return jsonify({"message": "User logged in with Google", "user_id": str(user['_id'])}), 200

# Complete registration endpoint
@app.route('/complete-registration', methods=['POST'])
def complete_registration():
    data = request.get_json()
    uid = data.get('uid')
    username = data.get('username')
    password = data.get('password')

    # Validate inputs
    if not uid or not username or not password:
        return jsonify({"error": "Missing uid, username, or password"}), 400

    # Hash the password before storing it in the database
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Find the user in the database
    user = users_collection.find_one({"google_id": uid})
    if user:
        # Update user record with username and password
        users_collection.update_one(
            {"google_id": uid},
            {"$set": {"username": username, "password": hashed_password.decode('utf-8')}}
        )
        return jsonify({"message": "Registration completed successfully"}), 200
    else:
        return jsonify({"error": "User not found"}), 404

# Protected Profile retrieval endpoint
@app.route('/profile', methods=['GET'])
def get_profile():
    google_token = request.headers.get("Authorization")
    idinfo = verify_google_token(google_token)

    if idinfo is None:
        return jsonify({"error": "Invalid Google token"}), 401

    google_id = idinfo['sub']
    user = users_collection.find_one({"google_id": google_id})

    if user:
        user_data = {
            "name": user.get('name'),
            "email": user.get('email'),
            "username": user.get('username'),
            "age": user.get('age'),
            "gender": user.get('gender'),
            "height": user.get('height'),
            "weight": user.get('weight')
        }
        return jsonify(user_data), 200
    else:
        return jsonify({"error": "User not found"}), 404


# Endpoint to receive and process image data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        serving_size = data.get("serving_size", 1)  # Default to 1 if not provided

        if not image_base64:
            return jsonify({"error": "No image data provided"}), 400

        # Log the input
        print("Received image data (base64):", image_base64[:100])  # Print only first 100 characters for debugging

        # Decode the base64 image
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            print("Base64 decoding error:", str(e))
            return jsonify({"error": f"Base64 decoding error: {str(e)}"}), 400

        # Log the decoded image bytes
        print("Decoded image bytes:", image_bytes[:20])  # Print first 20 bytes for debugging

        # Process the image
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            print("Image processing error:", str(e))
            return jsonify({"error": f"Cannot identify image file: {str(e)}"}), 400

        # Predict the class
        class_index = predict_image(image_bytes)

        # Food class labels
        food_classes = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"]  # Ensure this matches your model's output
        predicted_food = food_classes[class_index]
        formatted_food = format_food_name(predicted_food)

        # Retrieve calories
        calories_per_100g = calorie_map.get(predicted_food, "Unknown")
        if calories_per_100g != "Unknown":
            estimated_calories = float(calories_per_100g) * serving_size
        else:
            estimated_calories = "Unknown"

        return jsonify({
            "predicted_class": format_food_name(predicted_food),
            "calories": estimated_calories
        }), 200
    except Exception as e:
        print("Unexpected error:", str(e))  # Log the error
        return jsonify({"error": str(e)}), 500


# Endpoint to calculate calorie needs based on user profile and goal
@app.route('/user_calories', methods=['GET'])
def user_calories():
    google_token = request.headers.get("Authorization")
    idinfo = verify_google_token(google_token)

    if idinfo is None:
        return jsonify({"error": "Invalid Google token"}), 401

    google_id = idinfo['sub']
    user = users_collection.find_one({"google_id": google_id})
    if not user:
        return jsonify({"error": "User not found"}), 404

    height = user.get('height')
    weight = user.get('weight')
    age = user.get('age')
    gender = user.get('gender')

    if not all([height, weight, age]):
        return jsonify({"error": "Incomplete profile information"}), 400

    # Base calorie calculation using Mifflin-St Jeor Equation
    calories = 10 * weight + 6.25 * height - 5 * age
    if gender == 'female':
        calories -= 161
    else:
        calories += 5

    # Adjust calorie needs based on the goal
    goal = request.args.get("goal", "Maintain Weight")
    if goal == 'Maintain Weight':
        pass  # No change needed
    elif goal == 'Fat Loss':
        calories *= 0.8  # 20% deficit for weight loss
    elif goal == 'Weight Gain':
        calories *= 1.2  # 20% surplus for weight gain
    else:
        return jsonify({"error": "Invalid goal specified"}), 400

    return jsonify({"calories_needed": round(calories), "goal": goal}), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

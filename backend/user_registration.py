from flask import Flask, request, jsonify
from pymongo import MongoClient
import bcrypt
import google.auth.transport.requests
import google.oauth2.id_token

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb+srv://emrannooremon70:emon1234@nutritionapp.lecyp.mongodb.net/nutritionApp?retryWrites=true&w=majority')
db = client['nutritionApp']
users_collection = db['users']

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

@app.route('/google_signup', methods=['POST'])
def google_signup():
    data = request.get_json()
    google_token = data.get('google_token')
    name = data.get('name')
    email = data.get('email')

    try:
        # Verify the token
        id_info = google.oauth2.id_token.verify_oauth2_token(
            google_token, google.auth.transport.requests.Request()
        )
        google_id = id_info['sub']

        # Check if the user already exists
        user = users_collection.find_one({"google_id": google_id})
        if not user:
            # Create new user
            user_data = {
                "name": name,
                "email": email,
                "google_id": google_id
            }
            users_collection.insert_one(user_data)
            return jsonify({"message": "User signed up with Google"}), 201

        return jsonify({"message": "User logged in with Google"}), 200
    except ValueError:
        return jsonify({"error": "Invalid Google token"}), 400

if __name__ == "__main__":
    app.run(debug=True)
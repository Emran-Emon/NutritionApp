from pymongo import MongoClient
from bson.objectid import ObjectId

# MongoDB setup
client = MongoClient('mongodb+srv://emrannooremon70:emon1234@nutritionapp.lecyp.mongodb.net/nutritionApp?retryWrites=true&w=majority')
db = client['nutritionApp']
users_collection = db['users']

def create_user(name, email, username=None, password=None, google_id=None):
    user_data = {
        "name": name,
        "email": email,
        "password": password,
        "google_id": google_id,
        "username": username,
    }
    return users_collection.insert_one(user_data).inserted_id

def find_user_by_email(email):
    return users_collection.find_one({"email": email})

def find_user_by_google_id(google_id):
    return users_collection.find_one({"google_id": google_id})

def update_user_password(user_id, password):
    return users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password": password}}
    )

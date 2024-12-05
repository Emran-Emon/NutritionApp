from pymongo import MongoClient
from parse_food_names import parse_food_names

client = MongoClient('mongodb+srv://emrannooremon70:emon1234@nutritionapp.lecyp.mongodb.net/nutritionApp?retryWrites=true&w=majority')
db = client['nutritionApp']
foods_collection = db['foods']

food_names = parse_food_names()

def insert_foods():
    food_documents = [{"name": food_name} for food_name in food_names]
    result = foods_collection.insert_many(food_documents)
    print(f"{len(result.inserted_ids)} food items inserted into MongoDB.")

def insert_user(user_data):
    user_collection = db['users']
    user_collection.insert_one(user_data)

def update_user_info(user_id, new_data):
    user_collection = db['users']
    result = user_collection.update_one({"user_id": user_id}, {"$set": new_data})
    return result.modified_count > 0

def get_user_by_id(user_id):
    user_collection = db['users']
    user = user_collection.find_one({"user_id": user_id}, {"_id": 0})
    return user


if __name__ == "__main__":
    insert_foods()

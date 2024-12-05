import json
from pymongo import MongoClient

client = MongoClient('mongodb+srv://emrannooremon70:emon1234@nutritionapp.lecyp.mongodb.net/nutritionApp?retryWrites=true&w=majority')
db = client['nutritionApp']
collection = db['foods'] 

with open(r'E:\NutritionApp\kaggle\food-101\food-101\meta\test.json') as file:
    file_data = json.load(file)

if isinstance(file_data, list):
    collection.insert_many(file_data)
else:
    collection.insert_one(file_data)

print("Data imported successfully.")

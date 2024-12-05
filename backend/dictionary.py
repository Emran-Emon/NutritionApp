import requests
import time

API_KEY = 'I1VeR0OASlbfSLbDyGoDAL0KhXlvqpsBGeU2NQ8N'
BASE_URL = 'https://api.nal.usda.gov/fdc/v1/foods/search'

# List of Food-101 classes
food_classes = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", 
"cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", 
"sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"]

# Dictionary to store calorie mappings
calorie_map = {}

def get_calories_for_food(food_name):
    # Search for food and retrieve nutritional information
    params = {
        'api_key': API_KEY,
        'query': food_name.replace('_', ' '),  # Replace underscores with spaces
        'pageSize': 1  # Retrieve only the top search result
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        foods = data.get("foods")
        if foods:
            # Extract calorie info (per 100g or single serving)
            calories = foods[0].get("foodNutrients", [{}])[0].get("value")
            return calories
    return None  # Return None if no result or error

# Loop through all food classes to get calorie data
for food in food_classes:
    try:
        print(f"Fetching calorie data for: {food}")
        calories = get_calories_for_food(food)
        if calories:
            calorie_map[food] = calories
            print(f"  {food}: {calories} kcal")
        else:
            calorie_map[food] = "Unknown"
            print(f"  {food}: No calorie data found.")
        
        # Avoid hitting rate limits by pausing after each request
        time.sleep(1)
    except Exception as e:
        print(f"Error fetching data for {food}: {e}")
        calorie_map[food] = "Unknown"

# Save the calorie_map to a file for later use
import json
with open("calorie_map.json", "w") as f:
    json.dump(calorie_map, f)

print("Calorie mapping completed.")

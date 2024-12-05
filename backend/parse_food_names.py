def parse_food_names():
    food_names = []
    with open(r'E:\NutritionApp\kaggle\food-101\food-101\meta\classes.txt', 'r') as file:
        for line in file:
            food_name = line.strip()
            food_names.append(food_name)
    return food_names

if __name__ == "__main__":
    foods = parse_food_names()

    formatted_foods = '", "'.join(foods)
    formatted_string = f'["{formatted_foods}"]'

    print(formatted_string)

    with open('food_classes_formatted.txt', 'w') as file:
        file.write(formatted_string)

    print("Food classes have been parsed and saved in the desired format to 'food_classes_formatted.txt'.")
# Creating an AI App for Suggesting Diet for Different Ages
from helper_functions import get_llm_response, print_llm_response

user_name = input("Enter your name: ")
user_age_input = int(input("Enter Your Age: "))

list_of_type_diet = [
    "Healthy", 
    "Gym Diet", 
    "Weight Loss Diet", 
    "Weight Gain Diet",
    "Mediterranean Diet",
    "DASH Diet",
    "Anti-inflammatory Diet",
    "Plant-Based Diet",
    "Heart-Healthy Diet",
    "High-Protein Diet",
    "Ketogenic Diet",
    "Paleo Diet",
    "Low-Carb Diet",
    "Carb Cycling Diet",
    "Athlete Performance Diet",
    "Intermittent Fasting",
    "Low-Fat Diet",
    "Calorie Deficit Diet",
    "Weight Maintenance Diet",
    "Diabetic-Friendly Diet",
    "Gluten-Free Diet",
    "Dairy-Free Diet",
    "Low-Sodium Diet",
    "Renal Diet"
]

print(list_of_type_diet)
type_of_diet = input("Enter which type of diet you want: ")

records_of_peoples = []

if type_of_diet in list_of_type_diet:
    prompt = f"Suggest a full diet plan for {user_name}, who is {user_age_input} years old, and wants a {type_of_diet}."
    
    diet_plan = get_llm_response(prompt)
    
    records = {
        'name': user_name,
        'age': user_age_input,
        'diet_type': type_of_diet,
        'diet_plan': diet_plan 
    }
    
    records_of_peoples.append(records)
    
    print_llm_response(diet_plan) 
    
else:
    print("Invalid diet type.")

with open('diet_records.txt', 'w') as file:
    for record in records_of_peoples:
        file.write(f"Name: {record['name']}, Age: {record['age']}, Diet Type: {record['diet_type']}, Diet Plan: {record['diet_plan']}\n")

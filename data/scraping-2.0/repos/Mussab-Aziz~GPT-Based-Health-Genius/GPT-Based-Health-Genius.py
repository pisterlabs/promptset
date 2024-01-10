import openai
import json

# Function to make HTTP request to OpenAI API
def make_openai_request(prompt, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Function to calculate BMI
def calculate_bmi():
    weight = float(input("Enter your weight in kilograms: "))
    feet = float(input("Enter your height in feet: "))
    inches = float(input("Enter your height in inches: "))
    height_in_meters = (feet * 0.3048) + (inches * 0.0254)
    bmi = weight / (height_in_meters * height_in_meters)
    print(f'Your BMI is: {bmi}')
    return bmi

# Function to provide health advice using OpenAI
def health_advice(api_key, bmi):
    prompt = f"Provide health advice based on BMI {bmi}"
    response = make_openai_request(prompt, api_key)
    print(f'\nHealth Advice: {response}')

# Function to Set Goal using OpenAI
def set_goal(api_key, bmi):
    prompt = f"Suggest a fitness goal based on BMI {bmi}"
    response = make_openai_request(prompt, api_key)
    print(f'\nFitness Goal: {response}')

# Function to create a Diet Plan
def diet_plan(api_key, bmi):
    prompt = f"Suggest a Diet Plan of 3 meals based on BMI {bmi}"
    response = make_openai_request(prompt, api_key)
    print(f'\nDiet Plan: {response}')

# Function to Create a workout plan
def Workout_plan(api_key, bmi):
    prompt = f"Suggest a Workout plan based on BMI {bmi}"
    response = make_openai_request(prompt, api_key)
    print(f'\nWorkout Plan: {response}')

# Function to Create a meditation guide
def Meditation_Guide(api_key, bmi):
    prompt = f"Suggest a Meditation Guide based on BMI {bmi}"
    response = make_openai_request(prompt, api_key)
    print(f'\nMeditation Guide: {response}')

if __name__ == "__main__":
    api_key = "Your API Key"
    bmi = 0

    while True:
        # Menu
        print("\n*****\tHealth Care System\t*****\n")
        print("1- Calculate BMI ")
        print("2- Get health advice ")
        print("3- Set a Goal ")
        print("4- Create a diet plan ")
        print("5- Create a Workout plan ")
        print("6- Suggest a Medidation Guide plan ")
        choice = int(input("\nEnter your choice: "))

        if choice == 1:
            bmi = calculate_bmi()
        elif choice == 2:
            print("\nGenerating Health Advice....")
            health_advice(api_key, bmi)
        elif choice == 3:
            print("\nSetting a Goal....")
            set_goal(api_key, bmi)
        elif choice == 4:
            print("\nCreating a diet plan....")
            diet_plan(api_key, bmi)
        elif choice == 5:
            print("\nCreating a Workout plan....")
            Workout_plan(api_key, bmi)
        elif choice == 6:
            print("\nCreating a Medidation Guide ....")
            Meditation_Guide(api_key, bmi)
        else:
            print("Wrong Input.")
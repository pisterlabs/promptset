from decouple import config
import openai

openai.api_key = config('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')

def generate_meal_plan(name, gender, dob, bmi, height, weight, blood_glucose_level):
    prompt = f"Predict diabetic levels and suggest a meal plan for {name}, a {gender}, who has provided the following information:\n\n" \
             f"Name: {name}\nGender: {gender}\nDate of Birth: {dob}\nBMI: {bmi}\nHeight: {height}\nWeight: {weight}\n" \
             f"Blood Glucose Level: {blood_glucose_level}\n"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )

    meal_plan = response.choices[0].text.strip()
    return meal_plan

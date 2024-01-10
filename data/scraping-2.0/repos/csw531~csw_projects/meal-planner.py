import openai

def main(event, context):
    openai.api_key = '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'
    
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"Given your accumulated knowledge of health and nutrition, you are ideal for helping create my weekly meal plan. Please provide a 7-day meal plan tailored for a {event['age']} year old {event['gender']} who has a {event['activity-level']} lifestyle. I have the following allergies or cannot eat the following foods: {event['dietary-restrictions']}. If the space there following foods: was blank, then there are no restrictions. I aim to {event['specific-goal']}. The plan should focus on {event['preferred-diet']}. Over the 7-day period, I plan to cook {event['number-of-breakfasts']} breakfasts, {event['number-of-lunches']} lunches and {event['number-of-dinners']} dinners. Please also include room for {event['number-of-snacks']} snacks per day.",

      max_tokens=256,
      temperature=0.7
    )

    if 'choices' in response and len(response['choices']) > 0:
        return response['choices'][0]['text']
    return "There was an error communicating with OpenAI!"

if __name__ == "__main__":
    # Gather all required details for the event dictionary
    age = input("Enter your age: ")
    gender = input("Enter your gender (e.g., male, female, other): ")
    activity_level = input("Describe your activity level (e.g., sedentary, active, very active): ")
    dietary_restrictions = input("List any dietary restrictions or allergies (leave blank if none): ")
    specific_goal = input("What is your specific goal (e.g., lose weight, maintain weight, gain muscle): ")
    preferred_diet = input("What type of diet do you prefer (e.g., vegetarian, keto, Mediterranean): ")
    number_of_breakfasts = input("How many breakfasts do you plan to cook over the 7-day period? ")
    number_of_lunches = input("How many lunches do you plan to cook over the 7-day period? ")
    number_of_dinners = input("How many dinners do you plan to cook over the 7-day period? ")
    snacks_per_day = input(f"How many snacks per day based on your {activity_level} lifestyle? ")

    # Construct the event dictionary
    event = {
        'age': age,
        'gender': gender,
        'activity-level': activity_level,
        'dietary-restrictions': dietary_restrictions,
        'specific-goal': specific_goal,
        'preferred-diet': preferred_diet,
        'number-of-breakfasts': number_of_breakfasts,
        'number-of-lunches': number_of_lunches,
        'number-of-dinners': number_of_dinners,
        'number-of-snacks': snacks_per_day  # I've renamed this to be clearer
    }

    context = {}  # assuming you don't have any specific context information
    print(main(event, context))



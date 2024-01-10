
import openai
openai.api_key = 'INSERT OPENAI API KEY'

def get_recipe_suggestions(predicted_food_item, user_preferences, calorie_requirements):
    prompt = f"""Give me recipe suggestions for {predicted_food_item} that align with my dietary preferences: {user_preferences} and calorie requirements: {calorie_requirements}.
      Do not give links to external websites. The max tokens are set to 500 so be mindful of the length and try to be concise without ending the sentence abruptly. It should have the whole method of making a whole recipe in a cookbook format."""
    response = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        max_tokens=500,
        n=1,  # Number of recipe suggestions to generate
        stop=None,
        temperature=0.6
    )
    suggestions = [choice['text'].strip() for choice in response.choices]
    return suggestions
    

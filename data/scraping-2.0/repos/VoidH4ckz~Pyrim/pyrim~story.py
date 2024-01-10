#Pyrim-New/story.py
import openai

def generate_story():
    # Configure OpenAI API with your API key
    api_key = "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"
    openai.api_key = api_key

    # Define the prompt for the story
    prompt = "can you give me a 20 line story about traveling in the world of Pyrim?"

    # Generate a story using a chat model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the appropriate chat model
        messages=[
            {"role": "system", "content": "You are a brave adventurer."},
            {"role": "user", "content": prompt},
        ],
    )

    # Extract the generated story from the API response
    story = response['choices'][0]['message']['content']

    return story
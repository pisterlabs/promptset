import openai
from config import api_key


# Function to generate story using OpenAI GPT-4 API
def generate_story_with_gpt4_api(prompt):
    # Set up OpenAI API credentials
    openai.api_key = api_key  # Replace with your actual API key

    # Define parameters for story generation
    model_engine = 'text-davinci-003'  # Replace with the appropriate model engine
    max_tokens = 512  # Maximum number of tokens in the generated story

    # Generate story using OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        n=1,
        stop=None,
    )
    story = response.choices[0].text.strip()
    return story

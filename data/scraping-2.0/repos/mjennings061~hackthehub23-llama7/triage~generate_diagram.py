"""Generate a diagram from a given prompt using DALL-E."""
import os
import requests
import openai
from PIL import Image
from io import BytesIO

# OpenAI API setup.
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


def create_prompt(order_of_events):
    """Shorten the prompt to the first 1024 characters."""
    # Get the prompt for DALL-E
    MAX_PROMP_LENGTH = 1000
    user_story = """
    Generate a diagram of a car crash from an incident report, so that I can see the direction of movement and position of the cars involved.
    Requirements: Must show at least two cars on a diagram. Must have the direction of travel annotated for each car. There must be no text in the image.
    """

    # Create the prompt from the incident report and user story.
    prompt = f"""
    {user_story}

    Incident report: {order_of_events}
    """

    if len(prompt) > MAX_PROMP_LENGTH:
        # Use Chat to get the prompt down to the right size.
        print("Shortening prompt to 1000 tokens...")
        conversation = [
            {"role": "system", "content": "Your task is to shorten the prompt to 1000 tokens, but retain the meaning of the prompt."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can choose the most suitable engine
            # Define a list of messages
            messages=conversation,
            max_tokens=2000,  # Set a limit on the response length
            api_key=api_key
        )
        # Get the new prompt from the response.
        new_prompt = response["choices"][0]["message"]["content"]

    else:
        new_prompt = prompt

    return new_prompt


def get_generated_image(order_of_events):
    """Generate a diagram representing the order of events."""
    
    # Create the prompt.
    prompt = create_prompt(order_of_events)

    # Get DALL-E to generate the image.
    print("Generating image...")
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    result = response['data'][0]['url']
    print("Created image.")

    img = result
    return img


if __name__ == "__main__":
    transcript = """
    1. The user was driving on the main road an hour ago.
    2. Another vehicle pulled out from the right side of the user near a set of traffic lights.
    3. The user was driving at 30mph and was worried about missing their appointment.
    4. There were no injuries reported.
    5. The user's car sustained damage on the right side and the wheel is making a strange noise.
    6. The user's car is not drivable.
    """

    img = get_generated_image(order_of_events=transcript)
    img.show()

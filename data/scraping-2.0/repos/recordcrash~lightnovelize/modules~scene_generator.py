import configparser
import json

from openai import OpenAI

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize the OpenAI client
client = OpenAI(
    api_key=config['DEFAULT']['OPENAI_API_KEY']
)

DEFAULT_PROMPT = ("Oil painting of a sea captain with a rugged face, gray beard, and deep-set blue eyes, "
                  "standing on a wooden ship's deck. He wears a weathered tricorn hat and a dark blue coat. "
                  "Pointing towards the horizon, a large white whale breaches the water. The crew, men and "
                  "women of diverse descent with various hair colors and styles, hustle around him, readying "
                  "harpoons and adjusting sails. Some wear striped shirts, others don vests and bandanas.")


def generate_dalle_prompt_from_chapter_and_data(section: str, data: dict, local_mode: bool = False) -> str:
    """
    Uses GPT-4 to read a chapter, find a scene and generate an AI art prompt with it and the received data.
    """
    # System prompt from the saved file
    with open("./prompts/generate_image_prompt_system_prompt.txt", "r") as f:
        system_prompt = f.read()

    # User prompt with the data and the descriptions concatenated
    user_prompt = f"{data}\n\n{section}"

    if not local_mode:
        # Call to GPT-4
        print("Calling GPT-4 for AI prompt generation...")
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        response_content = completion.choices[0].message.content
        print("Response from GPT-4 with generated prompt:")
        print(response_content)
    else:
        response_content = DEFAULT_PROMPT
    return response_content

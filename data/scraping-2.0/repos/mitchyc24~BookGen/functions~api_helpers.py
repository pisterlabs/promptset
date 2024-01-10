import openai
import os

# Initialize the OpenAI API with your API key.
# It's recommended to use environment variables for security reasons.
openai.api_key = os.environ.get('OPENAI_API_KEY')

def generate_content(prompt: str, max_tokens: int = 150) -> str:
    """
    Uses the OpenAI API to generate content based on a given prompt.

    Args:
    - prompt (str): The input string to guide the content generation.
    - max_tokens (int, optional): The maximum length of the response.

    Returns:
    - str: The generated content.
    """
    # Create a chat message
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]

    # Query the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        max_tokens=max_tokens 
    )
    
    return response.choices[0].text.strip()

def generate_image_prompt(chapter_content: str) -> str:
    """
    Generate an image prompt based on the chapter content.
    This function can be enhanced based on specific requirements.

    Args:
    - chapter_content (str): The content of the chapter.

    Returns:
    - str: The generated image prompt.
    """
    # This is a basic implementation.
    # You may want to use specific sentences or themes from the chapter content 
    # to create a more detailed and guided image prompt.
    return f"Illustration representing a scene from: {chapter_content[:100]}..."

if __name__ == "__main__":
    # Test the functions
    print(generate_content("Once upon a time in a land far away..."))
    print(generate_image_prompt("Sarah walked into the dark forest, and she saw a mysterious light."))

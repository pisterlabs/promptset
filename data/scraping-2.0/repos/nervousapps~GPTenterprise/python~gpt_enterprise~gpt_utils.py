"""
\U0001F9E0
GPT utils
\U0001F9E0
"""
import os
import openai
import requests
from typing import Tuple, List, Generator


EMPLOYEE_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts", "employees")


def generate_text(
    system_prompt: str, user_prompt: str, temperature: float, model: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo-16k")
) -> Generator:
    """


    Args:
        system_prompt (str): Initialize the system with the given system prompt
        user_prompt (str): Assistant will try to give the best answer for the given user prompt
        model (str): OpenAI model to be used
        temperature (float): Temperature

    Returns:
        Generator: GPT response object
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            # Initialize GPT with system prompt
            {
                "role": "system",
                "content": system_prompt,  # + " Use less words as possible."
            },
            # Generate text relating to the user's prompt
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    return response


def generate_image(
    base_name: str,
    user_prompt: str,
    output_directory: str,
    system_prompt: str = "",
    nb_image: int = 1,
) -> Tuple[str, List[str]]:
    """
    Generate a prompt base on user_prompt and inject it to DALL-E
    to generate images.

    Args:
        system_prompt (str): Initialize the system with the given system prompt
        user_prompt (str): Assistant will try to give the best answer for the given user prompt
        base_name (str): Images' base name
        output_directory (str): Images' output directory
        nb_image (_type_): Number of image to generate

    Returns:
        list: Generated image names
    """
    # Ask ChatGPT a prompt to generate image with DALL-E
    with open(os.path.join(EMPLOYEE_PROMPTS_PATH, "dall_e_prompter.txt"), "r") as file:
        response = openai.ChatCompletion.create(
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo-16k"),
            messages=[
                # Initialize ChatGPT to be a helpful assistant but that it remains the employee
                {
                    "role": "system",
                    "content": f"{file.read()}"
                    + f" You are also {system_prompt} But keep in mind that {file.read()}"
                    if system_prompt
                    else "",
                },
                # Generate a subject
                {"role": "user", "content": f"SUBJECT {user_prompt}"},
            ],
        )

    # Create images, troncate prompt to 70 characters
    # to be sure it will be accepted by DALL-E
    image_response = openai.Image.create(
        prompt=response.choices[0].message.content[:70],
        n=nb_image,
        size="1024x1024",
    )

    generated_image_names = []

    # Download images
    for index, image in enumerate(image_response["data"]):
        img_data = requests.get(image["url"]).content
        img_name = f"{base_name}_{index}.jpg"
        img_path = os.path.join(output_directory, img_name)
        with open(img_path, "wb") as handler:
            handler.write(img_data)
            generated_image_names.append(f"./{img_name}")

    return response.choices[0].message.content, generated_image_names

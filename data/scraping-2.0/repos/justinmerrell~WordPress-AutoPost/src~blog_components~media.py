import os

import runpod
import openai

from dotenv import load_dotenv

load_dotenv()

# API Keys
runpod.api_key = os.environ.get("RUNPOD_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")


def generate_img_prompts(blog_title, blog_post, title_prompt, body_prompt):
    """
    Generates an image prompt based on the provided blog title and post using OpenAI API.

    :param blog_title: str, the title of the blog post.
    :param blog_post: str, the body of the blog post.
    :return: str, generated image prompt.
    """
    with open("src/prompts/image.txt", "r", encoding="UTF-8") as image_prompt_file:
        prompt = image_prompt_file.read()

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "assistant", "content": title_prompt},
            {"role": "system", "content": blog_title},
            {"role": "assistant", "content": body_prompt},
            {"role": "system", "content": blog_post},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def produce_image(image_prompt):
    """
    Generates an image using RunPod API based on the provided image prompt.

    :param image_prompt: str, the image prompt.
    :return: str, generated image URL.
    """
    with open("src/prompts/image_negative.txt", "r", encoding="UTF-8") as negative_prompt_file:
        negative_prompt = negative_prompt_file.read()

    endpoint = runpod.Endpoint("kandinsky-v2")

    run_request = endpoint.run({"prompt": image_prompt, "negative_prompt": negative_prompt})

    run_results = run_request.output()

    return run_results["image_url"]

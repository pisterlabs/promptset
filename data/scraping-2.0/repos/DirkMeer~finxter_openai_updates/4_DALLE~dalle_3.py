from decouple import config
from openai import OpenAI
from utils import image_downloader

client = OpenAI(api_key=config("OPENAI_API_KEY"))


def dalle_3(query: str):
    response = client.images.generate(
        model="dall-e-3",
        prompt=query,
        size="1024x1024",
        quality="hd",  # standard or hd
        n=1,
    )
    image_url = response.data[0].url
    print(response)
    image_downloader(image_url)
    return image_url


# dalle_3("a dragon flying in the night sky breathing fire, mystical and magical.")


def dalle_3_w_prompt_choice(query: str, basic_prompt=False):
    basic_request = """
    I NEED to test how the tool works with extremely simple prompts.
    DO NOT add any detail, just use it AS-IS:
    """
    final_query = query + basic_request if basic_prompt else query
    response = client.images.generate(
        model="dall-e-3",
        prompt=final_query,
        size="1024x1024",
        quality="standard",  # standard or hd
        n=1,
    )
    image_url = response.data[0].url
    print(response)
    print(f"Revised prompt: {response.data[0].revised_prompt}")
    image_downloader(image_url)
    return image_url


dalle_3_w_prompt_choice(
    "a dragon flying in the night sky breathing fire, mystical and magical.",
    basic_prompt=True,
)

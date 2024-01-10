from random import randint
from openai import OpenAI
from requests import get

# local imports
from configuration import (
    THUMBNAIL_QUANTITY,
    OPENAI_API_KEY_IMG,
    IMAGE_MODEL,
    OUTPUT_PATH,
)


def download_thumbnail(url: str, save_path: str) -> None:
    try:
        response = get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"Thumbnail downloaded successfully and saved at {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def generate_thumbnail(
    prompt: str,
    folder_name: str,
    pixels: int = 1024,
    model: str = IMAGE_MODEL,
    quality: str = "standard",
) -> None:
    print("Thumbnail generation...")
    client = OpenAI(api_key=OPENAI_API_KEY_IMG)

    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=f"{pixels}x{pixels}",
        quality=quality,
        n=THUMBNAIL_QUANTITY,
    )

    thumbnail_url = response.data[0].url
    if THUMBNAIL_QUANTITY > 1:
        download_thumbnail(
            url=thumbnail_url,
            save_path=f"{OUTPUT_PATH}\\{folder_name}\\thumbnail_{randint(1000000,9999999)}.png",
        )

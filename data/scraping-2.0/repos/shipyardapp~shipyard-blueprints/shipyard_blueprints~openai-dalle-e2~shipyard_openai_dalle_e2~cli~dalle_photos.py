import os
import openai
import requests


def main():
    api_credentials = os.environ.get("api_credentials")
    photo_prompt = os.environ.get("photo_prompt")
    number_of_photos = os.environ.get("number_of_photos")
    photo_topic = os.environ.get("photo_topic")

    openai.api_key = api_credentials
    response = openai.Image.create(
        prompt=photo_prompt, n=int(number_of_photos), size="1024x1024"
    )
    url_list = []
    for i in range(0, len(response["data"])):
        image_url = response["data"][i]["url"]
        url_list.append(image_url)

    for index, url in enumerate(url_list):
        destination_name = f"{photo_topic}_{index}.png"
        print(destination_name)
        with requests.get(url) as r:
            r.raise_for_status()
            with open(destination_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=(16 * 1024 * 1024)):
                    f.write(chunk)


if __name__ == "__main__":
    main()

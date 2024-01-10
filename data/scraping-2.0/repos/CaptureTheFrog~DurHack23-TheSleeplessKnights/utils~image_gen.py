import openai
import requests

from keys import openaikey


def gen_image(prompt, song_title):
    openai.api_key = openaikey

    generated_data = openai.Image.create(
        prompt=f"{prompt}",
        n=1,
        size="1024x1024",
        response_format="url",
    )

    filename = "static/images/" + song_title + ".jpg"

    generated_image_data = requests.get(generated_data["data"][0]["url"]).content
    with open(filename, 'wb') as handler:
        handler.write(generated_image_data)


def art_for_song(song_title):
    gen_image(f"Design an imaginative album cover that captures the essence of the song '{song_title}' by using "
              f"vibrant colors, surreal elements, and a blend of futuristic and organic textures. Consider "
              f"incorporating elements of nature, technology, and music into the artwork to evoke a sense of wonder "
              f"and creativity, reflecting the song's dynamic and evocative melody.'", song_title)

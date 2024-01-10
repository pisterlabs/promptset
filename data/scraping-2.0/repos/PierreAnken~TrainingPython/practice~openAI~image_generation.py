import os
import openai

if __name__ == '__main__':

    openai.api_key = os.environ.get("OPENAI_KEY")
    response = openai.Image.create(
        prompt="Logo cuisine",
        n=3,
        size="256x256"
    )
    print(response)


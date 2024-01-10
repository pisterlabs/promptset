from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe the contents of this photo.",
            },
            {
                "type": "image_url",
                # A photo of an airplane's instrument panel.
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Cessna_172_%282351649088%29_%282%29.jpg/640px-Cessna_172_%282351649088%29_%282%29.jpg",
            },
        ],
    },
]


def main():
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
    )
    print(response.choices[0].message)


if __name__ == "__main__":
    main()

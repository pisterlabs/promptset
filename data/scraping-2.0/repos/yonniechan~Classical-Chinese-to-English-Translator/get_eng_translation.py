from openai import OpenAI
import argparse
from apikey import api_key
client = OpenAI(api_key)

def translate(prompt, text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": f"{prompt}"
            },
            {
            "role": "user",
            "content": f"{text}"
            }
        ],
        temperature=0.7,
        # max_tokens=64,
        top_p=1
    )

    result = response.choices[0].message.content
    return result


def main(params):
    prompt = params.prompt
    text = params.text
    translated_text = translate(prompt, text)
    print(translated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    params = parser.parse_args()
    main(params)
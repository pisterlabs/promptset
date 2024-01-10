import os
import sys
import openai

# import codecs

# sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

if __name__ == "__main__":
    message = sys.argv[1]

    openai.api_key = os.environ.get("API_KEY")

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )

    print(completion.choices[0]["message"]["content"].strip())

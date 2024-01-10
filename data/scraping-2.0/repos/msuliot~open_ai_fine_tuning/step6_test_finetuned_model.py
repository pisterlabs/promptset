import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def main():
    ##### You will need to replace the FINE_TUNED_MODEL_ID with the one you got from the previous step.
    completion = openai.ChatCompletion.create(
        model="FINE_TUNED_MODEL_ID",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful and professional customer service representative"},
            {"role": "user", "content": "dude... i forgot my password."},
        ]
    )

    print(completion.choices[0].message)


if __name__ == "__main__":
    main()

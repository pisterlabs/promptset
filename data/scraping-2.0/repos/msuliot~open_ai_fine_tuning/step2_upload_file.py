import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def main():
    ft_file = openai.File.create(file=open("data.jsonl", "rb"), purpose='fine-tune')
    print(ft_file)
    print("Here is the training file id you need for Step 4 ==> ", ft_file["id"])


if __name__ == "__main__":
    main()

 
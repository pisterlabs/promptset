import openai
import datetime

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def pretty_table(f):
    print(f"\n{'ID':<33} {'Purpose':<20} {'Status':<12} {'Created At'}")
    print('-' * 88)
    for file in f['data']:
        created_at = datetime.datetime.fromtimestamp(file['created_at']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{file['id']:<33} {file['purpose']:<20} {file['status']:<12} {created_at}")


def main():
    file_list = openai.File.list(limit=25)
    # print(file_list)
    pretty_table(file_list)


if __name__ == "__main__":
    main()
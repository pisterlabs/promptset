from openai import OpenAI
import os
from dotenv import load_dotenv

def main():
  load_dotenv()

  client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
  )

  response = client.fine_tuning.jobs.list()

  print(response)

if __name__ == '__main__':
  main()

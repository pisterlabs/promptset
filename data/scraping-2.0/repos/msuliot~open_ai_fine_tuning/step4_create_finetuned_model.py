import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def main():
    ##### You will need to replace the TRAINING_FILE_ID with the one you got from the previous step.
    ft_job = openai.FineTuningJob.create(training_file="TRAINING_FILE_ID", model="gpt-3.5-turbo-0613")
    print(ft_job)  


if __name__ == "__main__":
    main()


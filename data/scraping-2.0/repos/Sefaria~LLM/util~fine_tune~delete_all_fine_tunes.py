import openai
from openai.error import TryAgain, InvalidRequestError
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    # would be nice to get models that exist and not jobs. look into openai.Models.
    fine_tunes = openai.FineTuningJob.list()
    for fine_tune in fine_tunes['data']:
        print('Deleting', fine_tune.fine_tuned_model)
        try:
            openai.Model.delete(fine_tune.fine_tuned_model)
        except (TryAgain, InvalidRequestError):
            print("skip")

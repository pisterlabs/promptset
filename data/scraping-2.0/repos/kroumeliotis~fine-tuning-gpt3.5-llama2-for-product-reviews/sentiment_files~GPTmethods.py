import os
import openai
from openai import OpenAI
import json


class GPTmethods:
    def __init__(self, model_id='gpt-3.5-turbo'):
        openai.api_key = os.environ.get("OPENAI_API_KEY")  # Access environment variable
        self.model_id = model_id
        # self.prompt = "Assign integer star ratings (between 1 and 5) to the following product reviews using the format: [1, 3, ...]. Please avoid providing additional explanations. Reviews:\n"
        self.prompt = 'Predict the star ratings (integer between 1 and 5) to the following product reviews. Return your response in json format like this example {"rating1":integer,"rating2":integer,...}. Please avoid providing additional explanations. Reviews:\n'

    """
    Create a conversation with GPT model
    """

    def gpt_conversation(self, conversation):
        client = OpenAI()
        # response = openai.ChatCompletion.create(
        completion = client.chat.completions.create(
            model=self.model_id,
            messages=conversation
        )
        return completion.choices[0].message
        # api_usage = response['usage']
        # print('Total token consumed: {0}'.format(api_usage['total_tokens']))
        # print(response['choices'][0].finish_reason)
        # print(response['choices'][0].index)
        # conversation.append(
        #     {'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
        # return conversation

    """
    Clean the response
    """

    def gpt_clean_response(self, conversation):
        try:
            # Ensure conversation is a dictionary
            if isinstance(conversation, dict):
                data = conversation
            else:
                # Parse conversation from a JSON string to a dictionary
                data = json.loads(conversation)

            # Extract all the numbers (ratings) from the dictionary
            ratings = []
            for key, value in data.items():
                if isinstance(value, int):
                    ratings.append(value)

            return ratings
        except json.JSONDecodeError as e:
            print("It is not valid JSON:", e)
        # # Parse the JSON-like string into a Python dictionary
        # data = json.loads(conversation)
        #
        # # Extract all the numbers (ratings) from the dictionary
        # ratings = []
        # for key, value in data.items():
        #     if isinstance(value, int):
        #         ratings.append(value)
        #
        # return ratings

    """
    Handle the response of GPT model
    """

    def gpt_ratings(self, reviews):
        if not isinstance(reviews, list):
            return {'status': False, 'data': 'Reviews variable is not a list'}
        else:
            my_prompt = self.prompt
            ii = 1
            for review in reviews:  # add the reviews into the prompt
                my_prompt += f"{ii}. \"{review}\"\n"
                ii += 1
            conversation = []
            conversation.append({'role': 'system', 'content': my_prompt})
            conversation = self.gpt_conversation(conversation)  # get the response from GPT model
            print(conversation)
            ratings = self.gpt_clean_response(conversation.content)
            if len(ratings) == len(reviews):
                return {'status': True, 'data': ratings}
            else:
                # return ratings
                return {'status': False,
                        'data': 'The ratings returned by the model do not match the number of reviews.' + '\n' + str(
                            ratings) + '\n' + str(reviews)}

    """
    Upload Dataset for GPT Fine-tuning
    """

    def upload_file(self, dataset):
        upload_file = openai.File.create(
            file=open(dataset, "rb"),
            purpose='fine-tune'
        )
        return upload_file

    """
    Train GPT model
    """

    def train_gpt(self, file_id):
        # https://www.mlq.ai/gpt-3-5-turbo-fine-tuning/
        # https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model?ref=mlq.ai
        return openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
        # check training status (optional)
        # openai.FineTuningJob.retrieve(file_id)

    """
    Delete Fine-Tuned GPT model
    """

    def delete_finetuned_model(self, model):  # ex. model = ft:gpt-3.5-turbo-0613:personal::84kpHoCN
        return openai.Model.delete(model)

    """
    Cancel Fine-Tuning
    """

    def cancel_gpt_finetuning(self, train_id):  # ex. id = ftjob-3C5lZD1ly5OHHAleLwAqT7Qt
        return openai.FineTuningJob.cancel(train_id)

    """
    Get all Fine-Tuned models and their status
    """

    def get_all_finetuned_models(self):
        return openai.FineTuningJob.list(limit=10)

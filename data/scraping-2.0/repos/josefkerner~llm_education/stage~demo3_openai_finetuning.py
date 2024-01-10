'''
This file will test finetuning on a specific dataset
'''
from typing import List, Dict
from model.gpt.gpt_chat import GPT_turbo_model
# open jsonl file
import jsonlines
import time
import openai
class OpenAIFinetuning:
    def __init__(self):
        pass

    def prepare_data(self):
        '''
        Will prepare data for finetuning
        :return:
        '''
        examples = []

        with jsonlines.open('data/rec_dataset.jsonl') as reader:
            data = [obj for obj in reader]
            for row in data:
                example_messages = {
                    "messages": [
                        {"role":"system","content":"You are a helpful assistant in human resources department"},
                        {"role":"user","content":row['prompt']},
                        {"role":"assistant","content":row['completion']},
                    ]
                }
                examples.append(example_messages)
        #save to jsonl file
        with jsonlines.open('data/finetuning_data_gpt3.jsonl', mode='w') as writer:
            writer.write_all(examples)

    def wait_until_done(self,job_id):
        events = {}
        while True:
            response = openai.FineTuningJob.list_events(id=job_id, limit=10)
            # collect all events
            for event in response["data"]:
                print(event)
            messages = [it["message"] for it in response.data]
            for m in messages:
                if m.startswith("New fine-tuned model created: "):
                    return m.split("created: ")[1], events
            time.sleep(10)

    def finetune(self):
        model = 'gpt-3.5-turbo-0613'
        response = openai.File.create(file=open("data/finetuning_data_gpt3.jsonl", "rb"), purpose="fine-tune")
        uploaded_id = response.id
        print("Dataset is uploaded")
        print("Sleep 30 seconds...")
        time.sleep(30)  # wait until dataset would be prepared
        response = openai.FineTuningJob.create(training_file=uploaded_id, model=model)
        print("Fine-tune job is started")
        ft_job_id = response.id
        new_model_name, events = self.wait_until_done(ft_job_id)
        with open("new_model_name.txt", "w") as fp:
            fp.write(new_model_name)
        print(new_model_name)

    def get_fine_tuned_model_name(self):
        with open("result/new_model_name.txt") as fp:
            return fp.read()
    def test_finetuning(self):
        '''
        Will test finetuning on a specific dataset
        :return:
        '''
        model_name = self.get_fine_tuned_model_name()
        cv = """
                John Doe is a Data Scientist with 3 years of experience.
                He has worked on multiple projects in the past.
                He has a Master's degree in Computer Science.
                He is proficient in Python, SQL, and Machine Learning.
                He knows how to use TensorFlow,PyTorch and Computer Vision libs.
                """
        role = """
                We are looking for a Data Scientist with 3 years of experience.
                The candidate should have a Master's degree in Computer Science.
                The candidate should be proficient in Python, SQL, and Machine Learning.
                The candidate should know how to use TensorFlow and PyTorch.
                """
        prompt = f"""
                Generate a recommendation if candidate a fit for the job role based on role requirements and candidate experience.
                If a candidate is not a fit for the job role, generate only string : "Not a fit".
                Candidate name: John Doe,
                Candidate experience: {cv}
                Job role requirements: {role}
                Recommendation:
                """
        message = [{"role": "user", "content": prompt}]
        self.llm = GPT_turbo_model(
            cfg={
                'model_name': model_name,
            }
        )
        self.llm.generate([message])

if __name__ == '__main__':

    OpenAIFinetuning = OpenAIFinetuning()
    #OpenAIFinetuning.prepare_data()
    #Finetune
    OpenAIFinetuning.finetune()
    #Prepare data for OpenAI
    #OpenAIFinetuning.test_finetuning()


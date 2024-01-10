from model import DatasetModel
import streamlit as st
import json
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
import time
from concurrent.futures import ThreadPoolExecutor

# DatasetViewModelの定義
class DatasetViewModel:
    def __init__(self):
        self.model = DatasetModel()
        self.api_key = self.model.api_key

    def on_save_api_key_clicked(self, api_key):
        self.api_key = api_key
        self.model.save_api_key(api_key)
        
    def generate_user_input_list(self, rules):
        template = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template("Rules:\n {rules}")])
        
        functions = [
            {
                "name": "generate_human_messages",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "human_message_list": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": """Create 10 input message data for LLM fine tuning and store them in a list.
The following variables should be taken into account when creating the input messages, and the data should be created so that the entropy is as large as possible.
Variables: length of the sentence, type of sentence, format of the sentence, content of the sentence, language of the sentence, speaker of the sentence""",
                        },
                    },
                    "required": ["human_message_list"],
                },
            },
        ]

        #api_key = self.model.load_api_key()
        api_key = st.session_state['loaded api key']
        llm = ChatOpenAI(openai_api_key=api_key, temperature=1.0, model_name="gpt-4", functions=functions, function_call={"name": "generate_human_messages"})
        result = llm(template.format_messages(rules=rules))
        additional_kwargs = result.additional_kwargs
        function_call = additional_kwargs.get('function_call', {})
        arguments_str = function_call.get('arguments', '')
        arguments_dict = json.loads(arguments_str)
        human_message_list = arguments_dict.get('human_message_list', [])

        return human_message_list
        
    def generate_single_dataset(self, message, system_message="", human_message=""):
        #api_key = self.model.load_api_key()
        api_key = st.session_state['loaded api key']
        chat = ChatOpenAI(openai_api_key=api_key, temperature=1.0, model_name="gpt-4")
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message),
        ])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        return chain.run(message)

    def generate_dataset(self, edited_human_message_list, system_message="", human_message=""):
        generated_dataset = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda msg: self.generate_single_dataset(msg, system_message, human_message),
                                   edited_human_message_list)
        for result in results:
            generated_dataset.append(result)
        return generated_dataset
    
    def show_contents(self, folder_name, file_name):
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            decoded_lines = [json.loads(line) for line in lines]
            return decoded_lines

    def save_dataset(self, dataset_name, dataset):
        folder_name = 'dataset_folder'

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        dataset_name = dataset_name.replace(' ', '_')
        with open(os.path.join(folder_name, f"{dataset_name}.jsonl"), "w") as file:
            for data in dataset:
                json_line = json.dumps(data)
                file.write(json_line + '\n')
        
    def fine_tune(self, dataset_name, model, suffix, n_epochs):
        upload_file = openai.File.create(
            file=open("dataset_folder/" + dataset_name, 'rb'),
            purpose='fine-tune'
        )

        st.write("準備をしています。30秒程度待ってください。")
        time.sleep(60)
        st. write("ファインチューニング開始")

        start_train = openai.FineTuningJob.create(
            training_file=upload_file["id"],
            model=model,
            suffix=suffix,
            hyperparameters={
                "n_epochs" : n_epochs
            },
        )
        return start_train

        
    def on_save_api_key_clicked(self, api_key):
        self.model.save_api_key(api_key)
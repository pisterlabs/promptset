
# Tools
import configparser
import json
import gradio as gr
import time
import random
import os

# Langchain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)

# OpenAI
from langchain.chat_models import ChatOpenAI

# Tiktoken
import tiktoken



# utils

def get_config(config_class, config_name):
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    config_result = config[config_class][config_name]
    return config_result

def get_random_number_eleven():
    random_number = random.randint(10000000000, 99999999999)
    return random_number

# AI_models.py
class AIModels:
    def __init__(self, model_name:str):
        AImodels_config = configparser.ConfigParser()
        AImodels_config.read('config.ini', encoding='utf-8')
        self.config = ''
        self.model_name = ''
        self.model = None
        self.type = ''

        if model_name == "gpt-3.5-turbo":
            self.type = 'chat'
            self.model_name = str(model_name)
            config = AImodels_config['OPENAI']
            self.api_key = config['gpt_3.5_turbo_openai_api_key']
            self.model = OpenAIModels(0, self.api_key)

        elif model_name == "gpt-4":
            self.type = 'chat'
            self.model_name = str(model_name)
            config = AImodels_config['OPENAI']
            self.api_key = config['gpt_4_openai_api_key']
            self.model = OpenAIModels(1, self.api_key)

        elif model_name == 'cl100k_base':
            self.type = 'tiktoken'
            self.model_name = str(model_name)
            self.model = TiktokenModels(0)

        elif model_name == 'p50k_base':
            self.type = 'tiktoken'
            self.model_name = str(model_name)
            self.model = TiktokenModels(1)

        elif model_name == 'r50k_base':
            self.type = 'tiktoken'
            self.model_name = str(model_name)
            self.model = TiktokenModels(2)

        else:
            # Model not Supported
            print(f"Model {model_name} is not supported!")


    def use_model(self, messages):
        if self.type == 'chat':
            # The model should have chat method
            response = self.model.chat(messages)
            return response
        elif self.type == 'tiktoken':
            if isinstance(messages,list):
                response = self.model.tik_token_messages(messages)
            elif isinstance(messages,str):
                sentence = messages
                response = self.model.tik_token_sentence(sentence)
            else:
                raise print(f"The input type is {str(type(messages))}, which is illegal!")
        else:
            raise print(f"Model {self.model_name} doesn't have this method!")


    def test_api_key(self):

        model_name = self.model_name
        model_api_key = self.api_key
        if model_name == "gpt-3.5-turbo":

            return 1
        elif model_name == "gpt-4":

            return 1
        else:

            return 2
        

class OpenAIModels:
    def __init__(self, index, api_key, max_length = 1000):
        self.openai_models = ['gpt-3.5-turbo', 'gpt-4']
        self.openai_model_name = self.openai_models[index]
        self.openai_api_key = api_key

    def chat(self, messages):
        chat = ChatOpenAI(model_name=self.openai_model_name, temperature=0, openai_api_key=self.openai_api_key, model_kwargs={"stop": ["\n"]})
        return_msg = chat(messages)
        response = return_msg.content
        return response

    def quick_test(self):
        """
            quick_test the validity of models.
        """
        test_text = "Give me just a word"
        result = ''
        chat = ChatOpenAI(model_name=self.openai_model_name, temperature=0, openai_api_key=self.openai_api_key)
        result = chat.predict(test_text)
        if result != '':
            return True
        else:
            return False

class TiktokenModels:
    def __init__(self, index):
        self.tiktoken_models = ['cl100k_base', 'p50k_base', 'r50k_base']
        self.tiktoken_model_name = self.tiktoken_models[index]
        self.encoding = tiktoken.get_encoding(self.tiktoken_model_name)

    def tik_token_messages(self, messages):
        num_of_tokens = len(self.encoding.encode(get_buffer_string(messages)))
        return num_of_tokens

    def tik_token_sentence(self, sentence):
        num_of_tokens = len(self.encoding.encode(str(sentence)))
        return num_of_tokens


# organizer.py

def fake_organizer(user_input, history):
    # 默认为assistant， character_id=69
    character_id = 69
    messages = [SystemMessage(content="You are an assistant.")]
    for i in history:
        messages.append(HumanMessage(content=str(i[0])))
        messages.append(AIMessage(content=str(i[1])))

    messages.append(HumanMessage(content=str(user_input)))
    return messages


def history_save_jsonl(history, save_path):
    dict_list = [{"Q":Q, "A":A} for Q,A in history]

    with open(save_path, 'w') as f:
        for entry in dict_list:
            json.dump(entry, f)
            f.write('\n')



class Map:
    def __init__(self, map_name):
        self.map_name = map_name
        self.map_id = get_random_number_eleven()
        self.map_npc_dict = {}
        self.map = {'map_name':self.map_name, 'map_id':self.map_id, 'map':[]}

        self.save_file_path = get_config('SAVE', 'map_save_file_path')
        self.map_file_path = os.path.join(self.save_file_path, self.map_name+str(self.map_id)+'.json')
        with open(self.map_file_path, 'w') as f:
            json.dump(self.map, f)

    def load_map_from_json(self, json_map):
        
        return
    
    def save_map(self):
        
        
        return self.map_id







import jsonlines
with jsonlines.open('map.jsonl', mode='w') as writer:
    writer.write({'map_name':"", 'map':[]}) 




m = Map("qianweiercun")























# gradio_server.py

with gr.Blocks(css=".gradio-container {background-color: red}") as demo:
    with gr.Row():
        with gr.Column(min_width=700):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="User Input")
            clear = gr.Button("Clear Chat")
            AI_model = AIModels("gpt-3.5-turbo")

            def user(user_message, history):
                return "", history + [[user_message, None]]
            
            def bot(history):
                print(history)
                user_input = history[-1][0]
                messages = fake_organizer(user_input, history[:-1])
                bot_message = []
                response = AI_model.use_model(messages)
                bot_message.append(response)
                history[-1][1] = ""
                for character in bot_message:
                    history[-1][1] += character
                    time.sleep(0.05)
                    yield history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.Column(min_width=700):
            img1 = gr.Image("db/odrusel_character/assistant_test/static/skeleton.png",
                            image_mode='L', 
                            show_download_button = False,
                            interactive = False, 
                            show_share_button = False,
                            show_label= False,
                            width=700, height=700)
            btn = gr.Button("Go")
    with gr.Row():
        text1 = gr.Textbox(label="t1")
        slider2 = gr.Textbox(label="s2")
        drop3 = gr.Dropdown(["a", "b", "c"], label="d3")
    
demo.queue()
demo.launch()
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from AI_models import AIModels
import configparser
import json

def input_preprocessing(user_input):
    return user_input

def fake_organizer(user_input, history):
    # 默认为assistant， character_id=69
    character_id = 69
    messages = [SystemMessage(content="You are an assistant.")]
    for i in history:
        messages.append(HumanMessage(content=str(i[0])))
        messages.append(AIMessage(content=str(i[1])))

    messages.append(HumanMessage(content=str(user_input)))
    return messages

def main_organizer(character_id: int, user_input: str, tiktoken_model: AIModels):
    organizer_config = configparser.ConfigParser()
    organizer_config.read('character_config.ini', encoding='utf-8')
    
    tiktoken_model.use_model()

    user_input = input_preprocessing(user_input)



    messages = ''
    return messages


def history_fetch(character_id):
    # 读取某个角色的history
    return

def history_memorizer():
    # 将history通过嵌入转换为memory向量
    return

def history_save_jsonl(history, save_path):
    dict_list = [{"Q":Q, "A":A} for Q,A in history]

    with open(save_path, 'w') as f:
        for entry in dict_list:
            json.dump(entry, f)
            f.write('\n')





#test
l = [['Who are you?', 'I am an AI assistant designed to help answer questions and assist with tasks. How can I assist you today?'], ['give me a complex english word.', 'One complex English word is "sesquipedalian," which means using long words or characterized by long words; long-winded.'], ["I'm Larry Zhang", 'Hello Larry Zhang! How can I assist you today?'], ['Who am I?', None]]
history_save_jsonl(l,"../odrusel_character/assistant_test/history/history.jsonl")

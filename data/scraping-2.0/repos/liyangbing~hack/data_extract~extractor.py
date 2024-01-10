# -*- coding: utf-8 -*-
from tqdm import tqdm
from langchain import OpenAI, ConversationChain
from langchain.prompts.prompt import PromptTemplate
from data_processing.text_processor import TextProcessor
from config.config import *


class Extractor():

    def __init__(self):
        self.gpt_name = "gpt-3.5-turbo"

    def get_prompt_template(self):
        input_v_list_holder = ["text"]
        prompt_template = """请抽取下面对话文本中主要的技术问题以及回复的解决方案, 
        要求: 
        ---将问题与解决方案匹配;
        ---过滤与技术无关的问题、闲聊以及感谢语;
        ---用规范的语言表示;
        ---尽量报证抽取的每个问题都有来自文本中的回复;
        ---只需要回复抽取的问答即可;
        ---抽取问答时每条聊天冒号之前的 wxid_* 可以代表对问答文本之间的关系;

        开始抽取!
        文本在***中间:
        ***
        {text}
        ***
        抽取的问答对:"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_v_list_holder,
        )
        return prompt

    def get_data(self, fpath):
        text_docs_processor = TextProcessor(1500, 10)
        docs = text_docs_processor.load_data(fpath)
        text_data = text_docs_processor.split_data(docs)
        text_docs_processor.display_info(text_data)
        return text_data

    def get_llm_model(self):
        llm = OpenAI(temperature=0)
        return llm

    def extract_qa_data(self, template, data, llm):
        message = template.format(text=data)
        conversation = ConversationChain(llm=llm)

        return conversation.predict(input=message)

    def save_data(self, save_path, answer):
        with open(save_path, 'a', encoding='utf-8') as f:
            f.writelines([item + '\n' for item in answer if item])


if __name__ == "__main__":
    set_env()
    set_openai()

    extract_data = Extractor()
    fpath = '../dataset/message.txt'
    text_data = extract_data.get_data(fpath)
    llm = extract_data.get_llm_model()
    prompt_template = extract_data.get_prompt_template()

    print(text_data)

    with open('extract_DP3_QA_test1.txt', 'a', encoding='utf-8') as f:
        for item in tqdm(text_data):
            message = extract_data.extract_qa_data(
                prompt_template, item.page_content, llm)
            f.writelines([item + '\n' for item in message if item])

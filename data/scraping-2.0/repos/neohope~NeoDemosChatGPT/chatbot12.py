#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, yaml
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI

'''
KGMemory通过知识图谱，提升对话效果
通过ConversationChain调用
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    llm = OpenAI(temperature=0)
    from langchain.prompts.prompt import PromptTemplate
    from langchain.chains import ConversationChain

    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
    If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

    Relevant Information:

    {history}

    Conversation:
    Human: {input}
    AI:"""
    prompt = PromptTemplate(
        input_variables=["history", "input"], template=template
    )
    conversation_with_kg = ConversationChain(
        llm=llm, 
        verbose=True, 
        prompt=prompt,
        memory=ConversationKGMemory(llm=llm)
    )

    conversation_with_kg.predict(input="Hi, what's up?")
    conversation_with_kg.predict(input="My name is James and I'm helping Will. He's an engineer.")
    conversation_with_kg.predict(input="What do you know about Will?")

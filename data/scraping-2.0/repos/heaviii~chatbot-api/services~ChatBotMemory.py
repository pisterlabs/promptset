import io
import threading
from typing import Any, Dict, List, Union
import os
import sys
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

import openai

from config import Config
from memory.ChromaMemory import ChromaMemory
from memory.RedisMemory import RedisMemory
from services.ThreadedGenerator import ChainStreamHandler, ThreadedGenerator
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
from prompt.prompts import MORE_LIKE_THIS_GENERATE_PROMPT


class ChatBotMemory:
    def streamChat(question, user_id):
        generator = ThreadedGenerator()
        threading.Thread(target=ChatBotMemory.askQuestion, args=(generator, user_id, question)).start()
        return generator
    
    def askQuestion(generator, user_id, question):
        try:
            memory = RedisMemory(user_id)

            batch_messages = ChatBotMemory().getbatchMessages(memory,user_id, question)

            chat = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager(
            [ChainStreamHandler(generator),StreamingStdOutCallbackHandler]), verbose=True, model="gpt-3.5-turbo-16k")
 
            result = chat.generate(messages=batch_messages)

            print("res_dict:---",result)
            aiMessage = result.generations[len(batch_messages)-1][0].text
            memory.add_user_message(question)
            memory.add_ai_message(aiMessage)
            #memory.clear()
            ChromaMemory().add_histroy_texts(user_id, question, aiMessage)

            return result

        finally:
            generator.close()

    def chattest(self, question, user_id):
        aiMessage = "小盒是小盒科技有限公司xiaohe.ai的智能客服，我可以回答你的问题，你可以问我任何问题，我会尽力回答你。"
        ChromaMemory().add_histroy_documents(user_id, question, aiMessage)
        res = ChromaMemory().get_histroy_documents(user_id, question)
        print(res)
        return res

    def chat(self, question, user_id):
        memory = RedisMemory(user_id)
        #return memory.get_messages(limit=3)
        batch_messages = self.getbatchMessages(memory, user_id, question)
        
        chat = ChatOpenAI(temperature=0)

        result = chat.generate(batch_messages)
        print(result)
        print(result.llm_output['token_usage'])
        aiMessage = result.generations[len(batch_messages)-1][0].text
        memory.add_user_message(question)
        memory.add_ai_message(aiMessage)
        #memory.clear()
        ChromaMemory().add_histroy_texts(user_id, question, aiMessage)
        
        return aiMessage
    
    def getbatchMessages(self, memory, user_id, question):
        prompt = question
        messageList  = memory.get_messages(limit=3)
        
        history_similarity_result = ChromaMemory().get_histroy_texts(user_id, question)

        #return history_similarity_result
        print("history_similarity_result-------",history_similarity_result)
        if len(history_similarity_result)>0:
            #humanMessage = HumanMessage(content=history_similarity_result.get("human_message"))
            #aiMessage = AIMessage(content=history_similarity_result.get("ai_message"))
            contents  = ".".join([content["human_message"] for content in history_similarity_result if content != None])
            #prompt = MORE_LIKE_THIS_GENERATE_PROMPT
            #prompt = prompt.format(prompt=question, original_completion=contents)
            messageHistory = "\n".join([ ("Human: {human_message}\n".format(human_message=message.content) if isinstance(message,HumanMessage) else "AI: {ai_message}\n".format(ai_message=message.content)) for message in messageList if message != None])
            prompt = """The following is a friendly conversation between a human and an AI. The human and the AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Related of conversation Human:
{history}
Current conversation:
{chat_history_lines}
Human: {input}
AI:"""
            prompt = prompt.format(input=question, history=contents, chat_history_lines=messageHistory)
            print("prompt:---",prompt)

        batch_messages = [
            [
                #SystemMessage(content="你是小盒科技的智能机器人,你是热情的,有创造力的,聪明的,友善的,有爱的."),
                HumanMessage(
                    content=prompt)
            ]
        ]
        #messageList.insert(0,systemMessage)
        #messageList.append(HumanMessage(content=prompt))
        #batch_messages = [memory.messages]

        print("batch_messages:---",batch_messages)
        
        return batch_messages
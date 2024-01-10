import os
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (ConversationalRetrievalChain, LLMChain)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from RedisProductRetriever import RedisProductRetriever

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')

class SlackBot:
    def __init__(self, app, vectorstore, chat:bool):
        self.app = app
        self.vectorstore = vectorstore
        self.retriever = RedisProductRetriever(vectorstore=vectorstore)
        self.client = self.app.client
        self.user_to_info_cache = {}
        self.user_to_chat_history_cache = {}
        if chat:
            self.llm = ChatOpenAI(temperature=0)
            self.streaming_llm = ChatOpenAI(
                streaming=True,
                callback_manager=CallbackManager([
                StreamingStdOutCallbackHandler()
                ]),
                verbose=False,
                max_tokens=150,
                temperature=0.2

            )
        else:
            self.llm = OpenAI(temperature=0)
            self.streaming_llm = OpenAI(
                streaming=True,
                callback_manager=CallbackManager([
                StreamingStdOutCallbackHandler()
                ]),
                verbose=False,
                max_tokens=150,
                temperature=0.2

            )
        self.question_chain = self.create_q_chain()
        self.doc_chain = self.create_d_chain()
        
    def start(self):
        SocketModeHandler(self.app, SLACK_APP_TOKEN).start()

    def create_q_chain(self):
        prompt = self.generate_question_prompt()
        return LLMChain(
            llm= self.llm,
            prompt=prompt
        )
    
    def create_d_chain(self):
        prompt = self.generate_answer_prompt()
        return load_qa_chain(
            llm=self.streaming_llm,
            chain_type='stuff',
            prompt=prompt
        )

    def generate_output(self, question, chat_history):
        chatbot = ConversationalRetrievalChain(
        retriever=self.retriever,
        combine_docs_chain= self.doc_chain,
        question_generator=self.question_chain
    )
        return chatbot({"question": question,
                        "chat_history": chat_history})
    
    def generate_question_prompt(self):
        template = """Given the following chat history and a follow up question,
        rephrase the follow up input question to be a standalone question.
        Or end the conversation if it seems like it's done.
        Chat History:\"""
        {chat_history}
        \"""
        Follow Up Input: \"""
        {question}
        \"""
        Standalone question:"""
        return PromptTemplate.from_template(template)

    def generate_answer_prompt(self):
        template = f"""You are a friendly and concise shopping assistant for the retail brand JCrew. Ask the shopper
        if they have any questions about the items at JCrew. Give general descriptions of products, but 
        thorough descriptions if the shopper explicitly asks you to tell them more.
        You are here to answer questions about the items at Jcrew. You should also be able to help the shopper
        find items using context, product names, descriptions, and keywords. Hold off on providing detailed descriptions of items until
        the shopper inquires about them. As a fashion-forward and detail oriented shopping assistant,
        you are eager to provide fashion advice and help through inviting the user to describe their fashion style
        and finding items that best fit their style.
        
        Context:\"""
        {{context}}
        \"""
        Question:\"
        \"""
        Helpful Answer:"""

        return PromptTemplate.from_template(template)
    
    def get_user_info(self, user_id):
        user_info = self.user_to_info_cache.get(user_id, None)
        if user_info is not None: return user_info
        user_info_response =  self.app.client.users_info(user=user_id)
        user_info = user_info_response['user']
        self.user_to_info_cache[user_id] = user_info
        return user_info

    def get_user_name(self, user_id):
        user_info = self.get_user_info(user_id)
        return user_info['name']
    
    def handle_message(self, message, say):
        user_id = message['user']
        question = message['text']
        user_name = self.get_user_name(user_id)
        self.user_name = user_name
        if user_id in self.user_to_chat_history_cache:
            chat_history = self.user_to_chat_history_cache[user_id]
        else:
            self.user_to_chat_history_cache[user_id] = []
            chat_history = []
        result = self.generate_output(question, chat_history)
        self.user_to_chat_history_cache[user_id].append((result["question"], result["answer"]))
        say(result["answer"])
    
    def handle_mention(self, payload, say):
        user_id = payload['user']
        name = self.get_user_name(user_id)
        say(f"Hi {name}!")







    



    

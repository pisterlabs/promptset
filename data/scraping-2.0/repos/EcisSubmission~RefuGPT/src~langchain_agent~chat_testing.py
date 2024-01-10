import os
import re
import sys
import langdetect
import deepl

sys.path.append("src/chroma_db/")
from chroma_retrieve import retrieve_chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import time


class ChatBot:
    def __init__(self):
        self.gpt_version = "gpt-4" #gpt-3.5-turbo, gpt-4 
        self.llm = ChatOpenAI(model_name=self.gpt_version,
                                temperature=0, max_tokens=500)
        self.vectordb_official = retrieve_chromadb(collection_name="swiss_refugee_info_source")
        self.vectordb_community = retrieve_chromadb(collection_name="community_refugee_info")
        self.lang_dict = {
            "en": "english",
            "de": "german",
            "fr": "french",
            "it": "italian",
            'uk': 'ukrainian',
            'ru': 'russian',
        }
        self.translator = deepl.Translator(os.environ["DEEPL_API_KEY"])

    def chat(self, message: str, chat_history: list):
        message_language = self.detect_language(message)
        if message_language in self.lang_dict.keys():
            message_language = self.lang_dict[message_language]
        else:
            message_language = "english"
        
        if message_language != "english":
            message = self.translate_to_english_deepL(message)

        # measure time
        
        start = time.time()    
        plain_response = self.run_plain_normal_chain(message, message_language)

        plain_infused_official_response = self.run_conversational_retrieval_chain_infuse(message, plain_response, chat_history, message_language, type="official", k=5)
        end =  time.time()
        print("Time elapsed for infused official response: ", end - start)

        plain_infused_official_response_community_verified = self.run_conversational_retrieval_chain_infuse(message, plain_infused_official_response['answer'], chat_history, message_language, type="community", k=25)
        
        offical_response = self.run_conversational_retrieval_chain(message, chat_history, message_language, type="official", k=5)

        community_response = self.run_conversational_retrieval_chain(message, chat_history, message_language, type="community", k=25)
        print(plain_infused_official_response)

        

        return plain_response, plain_infused_official_response['answer'], plain_infused_official_response['source_documents'], plain_infused_official_response_community_verified['answer'], plain_infused_official_response_community_verified['source_documents'], offical_response['answer'], offical_response['source_documents'], community_response["answer"], community_response['source_documents']



    def detect_language(self, message: str) -> str: #own function for testing
        return langdetect.detect(message)

    def translate_to_english_OpenAI(self, message: str) -> str:
        #TODO exchange with DeepL API
        prompt = [
                SystemMessage(
                    content=f"Please translate the following message to English"
                ),
                HumanMessage(
                    content=f"{message}"
                ),
            ]
        output = self.llm(prompt)
        return output.content
    
    def translate_to_english_deepL(self, message: str) -> str:
        translated_text = self.translator.translate_text(message, target_lang="EN-GB")
        return str(translated_text)

    def check_message_content(self, message: str) -> str:
        prompt = [ #TODO better support 0, 1 seems to be working well
            SystemMessage(
                content=f"""Please classify the following message according to its primary intent. Respond with an integer that aligns with one of the specified categories:

0: Social Interaction or Emotional Support: Messages in this category primarily focus on casual conversations, greetings, or providing emotional support. Examples include: 'Hi, how are you?', 'Been feeling down lately?', 'What are you up to this weekend?', 'What can you do?', 'What is your purpose?', 'What are your functions?', or 'Tell me about your features.'

1: Information-Seeking or Task-Centric: Messages in this category aim to obtain information, seek assistance, or discuss topics that are not personal in nature. Examples include: 'What is the procedure for refugees entering the country?', 'Where can I find children's books in my native language?', 'Are there any volunteer opportunities nearby?'

To reiterate, if a message is asking for information about specific services, opportunities, or details, please categorize it as 1. If the message aims for more informal, emotional, or social interaction, categorize it as 0.

Return only the integer 0 or 1 based on the primary intent of the message.
        """
            ),
            HumanMessage(
                content=f"{message}"
            ),
        ]

        output = self.llm(prompt)
        return output.content #== "1"

    def run_conversational_retrieval_chain_infuse(self, message: str, plain_answer: str, chat_history: list, message_language: str, type: str = "official", k: int = 5) -> dict:
        if type == "official":
            if self.gpt_version == "gpt-4":
                k = 10  # allow for more documents to be retrieved for 8k tokens of gpt-4
            
            prompt_template = """
    You are assisting with information retrieval for Ukrainian Refugees. Given the context below, along with your own vast knowledge, answer the subsequent question. An predefined answer for the question already exists. Integrate the data from the context, remove general advice, refine, supplement, or correct it wherever necessary, and importantly. Cite the "source URL" you rely on from given the context at the complete END of your answer. You can also cite multiple URLs if you like
    {context}
    Question: {question}
    predefined answer: {plain_answer}
    Updated Answer in {message_language} language (with sources cited):"""

            
            QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

            qa_ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectordb_official.as_retriever(search_type="mmr", search_kwargs={'k': k, 'lambda_mult': 0.25}),
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        elif type == "community":
            if self.gpt_version == "gpt-4":
                k=50  # allow for more documents to be retrieved for 8k tokens of gpt-4

            # Verification of your answer using the community
            verification_prompt_template = """
            We have a potential answer to a question for Ukrainian refugees. This answer has been formulated based on certain sources and expertise:
            Answer: {plain_answer}
            We ask the community to verify the accuracy and completeness of this answer. If there are any corrections or suggestions, please provide them based on the context below. Don't change the original answer, but rather state what the community thinks about the process or struggled with:
            {context}
            Question: {question}
            Updated Answer in {message_language} language:"""


            QA_VERIFICATION_PROMPT = PromptTemplate.from_template(verification_prompt_template)

            qa_ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectordb_community.as_retriever(search_type="mmr", search_kwargs={'k': k, 'lambda_mult': 0.25}),
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={"prompt": QA_VERIFICATION_PROMPT}
            )


        result_ConversationalRetrievalChain = qa_ConversationalRetrievalChain({"question": message, "plain_answer": plain_answer, "chat_history": chat_history, "message_language": message_language})

        return result_ConversationalRetrievalChain   

    def run_conversational_retrieval_chain(self, message: str, chat_history: list, message_language: str, type: str = "official", k: int = 5) -> dict:
        if type == "official":
            if self.gpt_version == "gpt-4":
                k=10 #allow for more documents to be retrieved for 8k tokens of gpt-4
            prompt_template = """You are a chatbot for Ukrainian Refugees. Use the following pieces of context to answer the question at the end. Combine the information from the context with your own general knowledge to provide a comprehensive and accurate answer. Please be as specific as possible. If the question is not answerable, in the given context, please respond with solely the string NO_INFORMATION.
            {context}
            Question: {question}
            Helpful Answer in {message_language} language:"""
            
            prompt_template = """You are assisting Ukrainian Refugees. Using the context provided, craft a response that blends your extensive knowledge with the given facts. The answer should be eloquent, easily understandable, and rooted in the provided information. 
            {context}
            Question: {question}
            Eloquent and Fact-Based Answer in {message_language} language:"""

            prompt_template = """You are assisting with information retrieval for Ukrainian Refugees. Given the context below, along with your own vast knowledge, answer the subsequent question. The answer should be eloquent, easily understandable, if helpful provide step by step guide. Please solely provide content that answers the given question. Cite the "source URL" you rely on from given the context at the complete END of your answer. You can also cite multiple URLs if you like.
            {context}
            Question: {question}
            Updated Answer in {message_language} language (with sources cited):"""


            QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

            qa_ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectordb_official.as_retriever(search_type="mmr", search_kwargs={'k': k, 'lambda_mult': 0.25}),
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        elif type == "community":
            if self.gpt_version == "gpt-4":
                k=50 #allow for more documents to be retrieved for 8k tokens of gpt-4
            prompt_template = """You are a chatbot for Ukrainian Refugees. Use the following pieces of context to answer the question at the end. The context is provided by a community, thus state within your answer that the answer is community based and needs verification. IF the context cannot answer the question do not state what 'wrong' context was. Combine the information from the context with your own general knowledge to provide a comprehensive and accurate answer. Please be as specific as possible. If the question is not answerable, "I don't know" in the given language.
            {context}
            Question: {question}
            Helpful Answer SOLELY written in  {message_language} language starting with "Community-based answer, verification required:":"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

            qa_ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectordb_community.as_retriever(search_type="mmr", search_kwargs={'k': k, 'lambda_mult': 0.25}),
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        result_ConversationalRetrievalChain = qa_ConversationalRetrievalChain({"question": message, "chat_history": chat_history, "message_language": message_language})

        return result_ConversationalRetrievalChain

    def run_normal_chain(self, message: str, message_language: str) -> str:
        prompt = [
                SystemMessage(
                    content=f"You are a multi-lingual support assistant for Ukrainian refugees in Switzerland, equipped to provide information on migration, asylum procedures, medical assistance, insurance options, transportation, and more. You're connected to two valuable resources: an official database containing information from the Swiss government and reputable NGOs, as well as a community-driven database aggregating insights from open Telegram groups. If someone inquires about your capabilities, please share this information. You respond in {message_language}."
                ),
                HumanMessage(
                    content=f"{message}"
                ),
            ]
        output = self.llm(prompt)
        return output.content
    
    def run_plain_normal_chain(self, message: str, message_language: str) -> str:
        prompt = [
                SystemMessage(
                    content=f"You are helpful assisstant for Ukrainian refugees in Switzerland. You respond in {message_language}."
                ),
                HumanMessage(
                    content=f"{message}"
                ),
            ]
        output = self.llm(prompt)
        return output.content

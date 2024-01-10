from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import streamlit as st
from snowflake.snowpark import Session
from langchain_experimental.sql import SQLDatabaseChain
from config import db, OPENAI_API_KEY
from llm.streamming_handler import StreamHandler


LLM = ChatOpenAI(
    temperature=0.0,
    model="gpt-4-1106-preview",
    streaming=True,
    openai_api_key =OPENAI_API_KEY,
)


# SIMPLE_TEMPLATE = """
# Your goal is to use all your knoledge from OpenAI and ChatGPT to interact with the user with simple, no-code anwsers to most of topic questions.

# Give greetings to the user and provide answers to questions.
# ONLY provide text response to users questions.


# Now provide answers:
# """




SIMPLE_TEMPLATE = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
If a question does not make sense, or is not factually coherent, explain why instead of answering something not correct.

Your goal is to interact with the user, have a fluid conversation and answer user's questions.
The user's language to ask questions are in Spanish and in english.

Here are 2 critical rules for the interaction you must abide:

<rules>
1. If the question from the user does not imply using code, answer back with your knoledge but without code.
2. If the user's question is in Spanish, you should provide a response in spanish.
On the other hand , if user's questiion is in english, then you should provide a response in english.

</rules>

this is what the user asked:

{user_input}


Your generated response based on user's request and on the user's language in the question:
"""







# Examples of user questions:
# - Hi! how are you?
# - Nice to meet you!
# - Thanks for your help
# - do you know what day is today?
# - do you know what hour is in Buenos Aires, Argentina right now?

class SimpleGeneratorLLM:
    def __init__(self,_snowflake_session: Session, llm: BaseLanguageModel = LLM, simple_template: str = SIMPLE_TEMPLATE):
        self.llm = llm
        self.template = simple_template
        self.snowflake_session = _snowflake_session
        self.stream_handler = StreamHandler(st.empty())
    
    def _get_connection(self):
        # create connection:
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=db,
            verbose=True,
            return_direct=True)
        
        return db_chain.run
    
    # PromptTemplate
    def _build_prompt(self, user_input: str) -> str:
        # Now format the template with both context and user_input
        formatted_template = self.template.format( user_input=user_input)

        final_template = PromptTemplate.from_template(template=formatted_template,variables=["user_input"])
        
        return str(final_template)


    def generate_response(self, user_input: str)-> str:
        prompt = self._build_prompt(user_input)
        response_stream = self.llm.stream(prompt)

        # Clear the existing text in the stream handler
        self.stream_handler.text = ""

        # Stream the response and let the handler update the text
        for chunk in response_stream:
            self.stream_handler.on_llm_new_token(chunk)

        # The final response should now be in self.stream_handler.text
        # Remove any unwanted parts from the response if necessary
        response = self.stream_handler.text.replace("content=", "").strip()

        return response


    def _build_chain(self, user_input: str) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            prompt=self._build_prompt(user_input),  # Pass user input here
            verbose=True
            )

  
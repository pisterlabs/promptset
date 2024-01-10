import os

import langchain
import pinecone
from langchain import ConversationChain, LLMChain
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
import openai
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.tools import Tool
from langchain.vectorstores import Pinecone

from utils.utils import set_openai_key


def set_langchain_key():
    set_openai_key()
    # print("key: ", openai.api_key)
    os.environ["OPENAI_API_KEY"] = openai.api_key


class LangChainService:
    def __init__(self, engine="gpt-3.5-turbo", max_tokens=180, n=1, stop=None, temperature=0.5):
        set_langchain_key()
        self.template_string = ""
        self.style = ""
        self.chat_openai_llm = ChatOpenAI(
            model_name=engine,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )
        # the following will not be supported by the new version of langchain
        self.openai_llm = OpenAI(
            model_name=engine,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )
        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)
        self.current_question = None
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True
        )
        self.conversation = ConversationChain(
            llm=self.chat_openai_llm,
            memory=self.memory,
            verbose=False
        )
        self.chain = LLMChain(
            llm=self.chat_openai_llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=False
        )
        self.picone_index = None
        self.picone_embed = None
        self.vectorstore = self.pinecone_init()
        self.qa = qa = RetrievalQA.from_chain_type(
                        llm=self.openai_llm,
                        chain_type="stuff",
                        retriever=self.vectorstore.as_retriever()
        )
        self.tools = [
            Tool(
                name='Knowledge Base',
                func=self.qa.run,
                description=(
                    'use this tool when answering general knowledge queries to get '
                    'more information about the topic'
                )
            )
        ]
        self.agent = initialize_agent(
            agent=ConversationChain,
            tools=self.tools,
            llm=self.chat_openai_llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self.memory
        )

    def pinecone_init(self):
        pinecone.init(
            api_key=os.getenv("PINECONE_KEY"),  # find at app.pinecone.io
            environment=os.getenv("PINECONE_ENVIRON"),  # next to api key in console
        )
        index_name = "vectordatabase"
        self.picone_index = pinecone.Index(index_name)
        self.picone_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)
        text_field = "text"

        return Pinecone(self.picone_index, self.picone_embed.embed_query, text_field)



    def set_langchain_template_question(self, text, format_instructions=None):
        self.reset_template_string()
        self.current_question = self.prompt_template.format_messages(
            style=self.style,
            text=text, format_instructions=format_instructions)

    def get_langchain_response(self, message=None):
        if self.current_question and message is None:
            message = self.current_question
            return self.chat_openai_llm(message)

        return self.openai_llm(message)

    # run this method before you call get_chain_prediction and get_langchain_response
    def reset_template_string(self):
        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)
        self.chain = LLMChain(
            llm=self.chat_openai_llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=False
        )

    def get_conversation_prediction(self, message):
        return self.conversation(message)

    def get_chain_prediction(self, prompt):
        return self.chain.run(prompt)


def sanity_test_template():
    langchain_service = LangChainService()
    langchain_service.template_string = """Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```"""
    langchain_service.style = """American English in a calm and respectful tone"""
    langchain_service.reset_template_string()
    customer_email = """Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, \
        the warranty don't cover the cost of \
        cleaning up me kitchen. I need yer help \
        right now, matey!
        """
    langchain_service.set_langchain_template_question(customer_email)
    print(langchain_service.current_question)
    print(langchain_service.get_langchain_response("How to learn ai tech quickly?"))
    print(langchain_service.get_langchain_response())

    print("Sanity Test Passed!")

def sanity_test_parser():
    langchain_service = LangChainService()
    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """

    langchain_service.template_string = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product \
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value

    text: {text}
    """

    langchain_service.reset_template_string()
    langchain_service.set_langchain_template_question(text=customer_review)
    response = langchain_service.get_langchain_response()
    print(response.content)

    gift_schema = ResponseSchema(name="gift",
                                 description="Was the item purchased\
                                 as a gift for someone else? \
                                 Answer True if yes,\
                                 False if not or unknown.")
    delivery_days_schema = ResponseSchema(name="delivery_days",
                                          description="How many days\
                                          did it take for the product\
                                          to arrive? If this \
                                          information is not found,\
                                          output -1.")
    price_value_schema = ResponseSchema(name="price_value",
                                        description="Extract any\
                                        sentences about the value or \
                                        price, and output them as a \
                                        comma separated Python list.")

    response_schemas = [gift_schema,
                        delivery_days_schema,
                        price_value_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    langchain_service.template_string = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product\
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    text: {text}

    {format_instructions}
    """
    langchain_service.set_langchain_template_question(text=customer_review, format_instructions=format_instructions)
    response = langchain_service.get_langchain_response()
    print(response.content)
    output_dict = output_parser.parse(response.content)
    print(output_dict)
    print("Sanity Test Passed!")


def sanity_test_conversation():
    langchain_service = LangChainService()
    print(langchain_service.conversation.predict(input="Hello, how are you?"))
    print("Sanity Test Passed!")


def sanity_test_chain():
    langchain_service = LangChainService()
    langchain_service.template_string = "What is the best name to describe \
        a company that makes {product}?"
    langchain_service.reset_template_string()
    print(langchain_service.chain.run("table"))
    print("Sanity Test Passed!")


if __name__ == "__main__":
    # sanity_test_template()
    # sanity_test_parser()
    # sanity_test_conversation()
    sanity_test_chain()

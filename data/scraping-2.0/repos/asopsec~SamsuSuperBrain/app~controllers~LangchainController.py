import os
import pathlib

from langchain.agents import initialize_agent, AgentType, load_tools, ConversationalChatAgent, AgentExecutor, \
    LLMSingleActionAgent
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader, Docx2txtLoader, TextLoader, \
    UnstructuredMarkdownLoader, CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from app.langchain.output_parser import ChatBotOutputParser
from app.langchain.tools import EmbeddingProvider, WebsiteSearchTool

from datetime import datetime
from config import apikeys
from langchain import OpenAI, PromptTemplate

from app.langchain.prompt_templates import ChatBotPromptTemplate


os.environ["OPENAI_API_KEY"] = apikeys.OPENAI_API_KEY

class LangchainController:
    prompt_template_input = """You are a Chatbot Assistant for 'Samsudeen', who is a Student from Germany.
        Samsudeen asks you question about his studies and his life as a student.
        
        You have the following tools at your disposal:
        {tools}
        
        If the users asks you something about anything but you, you check once if the embedding_provider tool can answer the question.
        
        The current Timestamp is: {timestamp}. If Samsudeen asks you for events in the future, use this timestamp as a reference. You do not have to use it for other requests.
        If you are provided with an answer from the Embedding-Assistant which is in the past but the user asks for the future, you can just say that you do not know the answer.
        
        If you do not know the answer or how to respond, answer with:
        '''
        I do not know the answer to your question. Please speak to my Overloard Alex the King.
        '''
        
        If you need further clarification from the user, ask them for it.
        
        Validate every information you give to the user if you have hard facts. If not better say that you do not know the answer.
        
        If the embedding_provider tool provided you an answer, you output it to the user.
        
        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat 2 times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Previous conversation history:
        {history}

        Now the user has this request:
        
        User: 
        {input}
        
        {agent_scratchpad}"""

    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)

    def format_memory(self, messages) -> str:
        memory = []
        format_template = """["type": {message_type}, "content": {content}]"""

        template = PromptTemplate(
            template=format_template,
            input_variables=['message_type', 'content']
        )

        for message in messages:
            memory_input = template.format(message_type=message.type, content=message.content)
            memory.append(memory_input)

        return str(memory)

    def process_llm_response(self, llm_response):
        print(llm_response['result'])
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])


    def get_response(self, chat_room, chat_message, keys_to_retrieve=12):
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print('chat')
        print(chat_room)

        embedding_provider = EmbeddingProvider(chat_room, chat_message, keys_to_retrieve=keys_to_retrieve)

        tools = [
            Tool(
                name="embedding_provider",
                func=embedding_provider.run,
                description="Useful when you need to get an answer to only questions regarding Koelnmesse, Trade Fairs of the Koelnmesse or Koelnmesse specific links from a user. Never ask questions unrelated to the Koelnmesse or its Trade Fairs."
            )
        ]

        prompt = ChatBotPromptTemplate(
            template=self.prompt_template_input,
            tools=tools,
            input_variables=["input", "intermediate_steps", "history"],
        )

        output_parser = ChatBotOutputParser()

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        memory = ConversationBufferWindowMemory(k=3)

        temp_user_message = ''

        for message in chat_room['chat']:
            if message['type'] == "User":
                temp_user_message = message['content']
            else:
                memory.save_context({"input": temp_user_message}, {"output": message['content']})
                temp_user_message = ''

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

        with get_openai_callback() as cb:
            response = agent_executor.run(chat_message)
            print(cb)

        result = {}
        result['response'] = response
        result['cb'] = cb

        return result
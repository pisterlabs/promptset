import os
import boto3
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import time
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name='gpt-3.5-turbo'
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)


class DataFetchingTool(BaseTool):
    name = "Workspace Data Fetcher"
    description = ("use this tool to get data from the workspace also referred to as private data or company data")

    def _run(self, query: str):
        s3 = boto3.client('s3')
        try:
            s3.download_file('mailqa-bucket', 'all_texts.txt', "all_texts.txt")
            print(f"File {'all_texts.txt'} downloaded successfully from {'mailqa-bucket'}")
            with open('all_texts.txt', 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            print(f"Error downloading {'all_texts.txt'} from {'mailqa-bucket'}: {e}")

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class EmailFetchingTool(BaseTool):
    name = "Email Data Fetcher"
    description = ("use this tool to get a users email data and inbox, do not use it if query involves company data")

    def _run(self, query: str):
        s3 = boto3.client('s3')
        try:
            s3.download_file('mailqa-bucket', 'emails.txt', "emails.txt")
            print(f"File {'emails.txt'} downloaded successfully from {'mailqa-bucket'}")
            with open('emails.txt', 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            print(f"Error downloading {'emails.txt'} from {'mailqa-bucket'}: {e}")

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")



tools = [ DataFetchingTool(), EmailFetchingTool()]

sys_msg = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# prompt = input(">>>")
# start_time = time.time()
# agent.run(prompt)
# end_time = time.time()
# duration = end_time-start_time
# print(duration)

# agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method='generate',
#     memory=conversational_memory
# )

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt

# update the agent tools
agent.tools = tools
prompt = input(">>>")
start_time = time.time()
agent.run(prompt)
end_time = time.time()
duration = end_time-start_time
print(duration)
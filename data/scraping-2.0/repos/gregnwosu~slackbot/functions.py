from dotenv import find_dotenv, load_dotenv
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.chains import LLMChain

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#https://www.youtube.com/watch?v=_pZebYlgGcY


load_dotenv(find_dotenv())


def draft_email(user_input, name="Dave"):
    chat = OpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a helpful assistant that drafts an email reply based on an a new email.
    
    Your goal is to help the user quickly create a perfect email reply.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    
    Make sure to sign of with {signature}.
    
    """

    signature = f"Kind regards, \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response



def more_clever(user_input):
    gmail_toolkit = GmailToolkit()
    llm = OpenAI(temperature=0)
    agent = initialize_agent(llm=llm, tools=gmail_toolkit, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
    response = agent.run(user_input)
    return response


# The AgentType enum defines the different types of agents that can be created. The following are the descriptions of each type of agent:

# ZERO_SHOT_REACT_DESCRIPTION - This type of agent can generate text in response to a prompt, even if it has never seen the prompt before.
# REACT_DOCSTORE - This type of agent can generate text in response to a prompt, and it can also access and use information from a document store.
# SELF_ASK_WITH_SEARCH - This type of agent can generate text in response to a prompt, and it can also ask itself questions and search for information to answer those questions.
# CONVERSATIONAL_REACT_DESCRIPTION - This type of agent can have a conversation with a user, and it can generate text in response to the user's questions and prompts.
# CHAT_ZERO_SHOT_REACT_DESCRIPTION - This type of agent can have a conversation with a user, and it can generate text in response to the user's questions and prompts, even if it has never seen the user before.
# CHAT_CONVERSATIONAL_REACT_DESCRIPTION - This type of agent can have a conversation with a user, and it can generate text in response to the user's questions and prompts, and it can also access and use information from a document store.
# STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION - This type of agent can have a conversation with a user, and it can generate text in response to the user's questions and prompts, and it can also access and use information from a document store, and it can generate text in a structured format.
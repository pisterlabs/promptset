import langchain
from dotenv import load_dotenv

# Load environnement variables, in particular OpenAI api key
load_dotenv()  

# Choose the model that will be used
chat = langchain.chat_models.ChatOpenAI(temperature=0.2, model_name='gpt-4',)


def get_basic_chat_chain(system_template: str, user_template: str) -> langchain.LLMChain:
    """
    Returns a langchain chain given an user and chat inputs

    Parameters:
        system_template (str): The system template
        user_template (str): The user template

    Returns:
        langchain.LLMChain: The langchain chain
    """

    system_prompt = langchain.prompts.SystemMessagePromptTemplate.from_template(
        system_template)

    user_prompt = langchain.prompts.HumanMessagePromptTemplate.from_template(
        user_template)

    chat_prompt = langchain.prompts.ChatPromptTemplate.from_messages([
        system_prompt,
        user_prompt
    ])

    return langchain.LLMChain(llm=chat, prompt=chat_prompt)
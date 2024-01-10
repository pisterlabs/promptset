from langchain.chat_models import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


short_term_memory_model = ChatOpenAI(
    temperature=0.5, 
    model='gpt-3.5-turbo'
)

malia_model = ChatOpenAI(
    temperature=0.9, 
    model='gpt-4-0613', 
    max_tokens=512
)


chuck_summary_model = ChatOpenAI(
        temperature=0,
        max_tokens=1000,
        model = 'gpt-3.5-turbo-0613'
    )

# Tweets generator
advanced_summary_model = ChatOpenAI(
    temperature=0.5,
    model='gpt-4-0613',
    request_timeout=120
)

malia_thought_model = ChatOpenAI(
    temperature=0.9, 
    model='gpt-4-0613',
    max_tokens=25
)


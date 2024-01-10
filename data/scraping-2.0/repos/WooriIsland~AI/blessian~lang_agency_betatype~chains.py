from pathlib import Path
import os

import dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from lang_agency_betatype import prompts

dotenv_file = dotenv.find_dotenv(str(Path("./").absolute().joinpath(".env")))
dotenv.load_dotenv(dotenv_file)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# llm = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

specifier_chain = LLMChain(
    prompt=prompts.specifier_chain_prompt,
    llm=llm,
    verbose=False,
)

conversation_chain = LLMChain(
    prompt=prompts.conversation_chain_prompt,
    llm=llm,
    verbose=False,
)
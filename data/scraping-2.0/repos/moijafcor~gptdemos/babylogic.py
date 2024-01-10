import sys
import os
import dotenv
from langchain_experimental.pal_chain.base import PALChain
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI

dotenv.load_dotenv()

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# llm = OpenAI(model_name='code-davinci-002',
#              temperature=0,
#              max_tokens=512)

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature":0.8}
    )

pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question = "Aspen has three times the number of pets as Audrey. Audrey has two more pets than Lara. If Lara has four pets, how many total pets do the three have?"

pal_chain.run(question)

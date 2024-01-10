from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from question_generation import generate_llama_prompt

# type true for gpt, type false for llama
def question_generation(llm, post, type:bool):
    system_prompt = ""
    
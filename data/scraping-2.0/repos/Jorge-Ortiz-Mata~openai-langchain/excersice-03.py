# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generic_answer(answer):
  llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo")

  prompt_template = PromptTemplate(
    input_variables=["answer"],
    template="{answer}"
  )

  name_chain = LLMChain(llm=llm, prompt=prompt_template)

  response = name_chain({"answer": answer})

  return response

question_one = generic_answer("Hello, who won the world cup in 2022?")

print(question_one)

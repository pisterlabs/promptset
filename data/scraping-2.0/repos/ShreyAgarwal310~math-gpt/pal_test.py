from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import PALChain
import openai
import pal
from pal.prompt import math_prompts

def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

api_key = get_file_contents('api_key.txt')

openai.api_key = api_key
OPENAI_API_KEY = api_key

interface = pal.interface.ProgramInterface(
  model='text-davinci-003',
  stop='\n\n\n', # stop generation str for Codex API
  get_answer_expr='solution()' # python expression evaluated after generated code to obtain answer 
)

question = 'Bob says to Alice: if you give me 3 apples and then take half of my apples away, then I will be left with 13 apples. How many apples do I have now?'

prompt = math_prompts.MATH_PROMPT.format(question=question)
answer = interface.run(prompt)
print(answer)

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, openai_api_key=api_key)
# palchain = PALChain.from_math_prompt(llm=llm, verbose=True)
# palchain.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")
# print(palchain.prompt.template)
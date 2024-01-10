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

interface = pal.interface.ProgramInterface(
  model='text-davinci-003',
  stop='\n\n\n', # stop generation str for Codex API
  get_answer_expr='solution()' # python expression evaluated after generated code to obtain answer 
)

# def solution():
#     "Bob says to Alice: if you give me 3 apples and then take half of my apples away, then I will be left with 13 apples. How many apples do I have now?"
#     apple_given = 3
#     apple_left = 13
#     apple_now = (apple_left + apple_given) * 2
#     result = apple_now
#     return result

# print(solution())

question = 'Bob says to Alice: if you give me 3 apples and then take half of my apples away, then I will be left with 13 apples. How many apples do I have now?'

prompt = math_prompts.MATH_PROMPT.format(question=question)
llm = OpenAI(model_name="text-davinci-003", temperature=0.9, openai_api_key=api_key)
palchain = PALChain.from_math_prompt(llm=llm, verbose=True)
palchain.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")
answer = interface.run(prompt)
print(answer)
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
style = """American English \
in a calm and respectful tone
"""
template_string = """Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

template = PromptTemplate(template=template_string, input_variables=["style","customer_email"])
chain = LLMChain(llm=llm, prompt=template)
response = chain.predict(style=style, customer_email=customer_email)

print(response)




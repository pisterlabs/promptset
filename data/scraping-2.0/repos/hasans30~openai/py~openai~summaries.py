from langchain.llms import OpenAI
from langchain import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv() 

openai_api_key=openai_api_key = os.getenv('OPENAI_API_KEY','no api key found');


# Note, the default model is already 'text-davinci-003' but I call it out here explicitly so you know where to change it later if you want
llm = OpenAI(temperature=0, model_name='text-davinci-003', openai_api_key=openai_api_key)

# Create our template
template = """
%INSTRUCTIONS:
Please summarize the following piece of text.
Respond in a manner that a 5 year old would understand.

%TEXT:
{text}
"""

# Create a LangChain prompt template that we can insert values to later
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

confusing_text = """
For the next 130 years, debate raged.
Some scientists called Prototaxites a lichen, others a fungus, and still others clung to the notion that it was some kind of tree.
“The problem is that when you look up close at the anatomy, it’s evocative of a lot of different things, but it’s diagnostic of nothing,” says Boyce, an associate professor in geophysical sciences and the Committee on Evolutionary Biology.
“And it’s so damn big that when whenever someone says it’s something, everyone else’s hackles get up: ‘How could you have a lichen 20 feet tall?’”
"""

print ("------- Prompt Begin -------")

final_prompt = prompt.format(text=confusing_text)
print(final_prompt)

print ("------- Prompt End -------")

output = llm(final_prompt)
print (output)
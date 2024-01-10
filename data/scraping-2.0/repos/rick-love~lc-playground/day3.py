from OpenAI_Training.config import get_api_key
from openai import OpenAI
from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.document_loaders import HNLoader
from langchain.document_loaders import UnstructuredURLLoader

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_api_key()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
llm = OpenAI(temperature=0.9, model_name="text-davinci-003")
chat = ChatOpenAI(temperature=.7)




############
# Document Loaders
loader = HNLoader("https://news.ycombinator.com/item?id=34422627")




############
# Show to the screen
# App Framework
import streamlit as st
st.title('Day 3:')
st.write()






############
# Output Parsers
# How you would like your response structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# How to parse the output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
output_parse_template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:

"""

output_parse_prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=output_parse_template
)

promptValue = output_parse_prompt.format(user_input="whenn is this going end")

llm_output_parser = llm(promptValue)
parsed = output_parser.parse(llm_output_parser)
# {
#   "bad_string": "whenn is this going end",
#   "good_string": "When is this going to end?"
# }






#################
# Example Selectors
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(), Chroma, k=2
)

simlar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,

    # Your prompt
    example_prompt=example_prompt,

    prefix="Give the location an item is usually found",
    suffix="Input: {noun}\nOutput",

    input_variables=["noun"]

)
my_noun = "plant"
# print(simlar_prompt.format(noun=my_noun))



#################
# Text Embeddings

embeddings = OpenAIEmbeddings()
text = "Learning AI is painful to start"
text_embedding = embeddings.embed_query(text)
# print(f"Embeddings Length: {len(text_embedding)}")
# print(f"A sample {text_embedding[2]}")


################
# Prompt Template
template = """
I really want to travel to {location}. What should I do there?

Respond with one short answer.

"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template
)

final_prompt = prompt.format(location="Lisbon")
# print(f"Final Prompt: {final_prompt}")
# print(f"LLM Output: {llm(final_prompt)}")




# Simple Chat
llm_response = llm('What day is after Tuesday')
response = chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like chicken, what should I eat?")
    ]
)


















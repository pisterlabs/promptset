from langchain import LLMChain, FewShotPromptTemplate
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = OpenAI(
    llm=llm,
    model="text-davinci-003",
    temperature=0
)



examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"}
]

example_template = """
Animal: {animal}
Habitat: {habitat}
"""

example_prompt = PromptTemplate(
    input_variables=["animal", "habitat"],
    template=example_template
)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal: {input}\nHabitat:",
    input_variables=["input"],
    example_separator="\n\n",
)

# Create the LLMChain for the dynamic_prompt
chain = LLMChain(
    llm=llm, 
    prompt=dynamic_prompt
)

# Run the LLMChain with input_data
input_data = {"input": "tiger"}
response = chain.run(input_data)

print(response)



# Additionally, you can also save your PromptTemplate to a file in your local filesystem in JSON or YAML format:

prompt_template.save("awesome_prompt.json")

# And load it back:

from langchain.prompts import load_prompt
loaded_prompt = load_prompt("awesome_prompt.json")

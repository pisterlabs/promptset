import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = "API_KEY"
# Text model example

llm = OpenAI(temperature=0.3)

template = """
    Search exactly price, current owner, creator information for {name} NFTs on {name_place} marketplace.
"""

prompt_template = PromptTemplate(
    input_variables=["content", "style"],
    template=template,
)

# Print the template after format
print(prompt_template.format(name="Nexian Gem", name_place="rarible"))

# Save prompt to json
prompt_template.save("Info.json")

# # Define a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

print(chain.run(name="Nexian", name_place="rarible"))

# First, create the list of few shot examples.
examples = [
    {
        "name": "LENA",
        "price": "$10.00",
        "owner": "Khôi",
        "creator": "albert herz", 
        "location": "niftygateway",
    },
    {
        "name": "LENA",
        "price": "$3,120",
        "owner": "sikoslovake",
        "creator": "Ies", 
        "location": "rarible",
    }
]

example_formatter_template = """
    Input name from user: {name}
    The information extracted from above command::\n
    ----
    Price: {price}\n
    Current owner: {owner}\n
    Creator: {creator}\n
    MarketPlace: {location}\n
"""

example_prompt = PromptTemplate(
    input_variables=["name", "price", "owner", "creator", "location"],
    template=example_formatter_template,
)


few_shot_prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    suffix="Input command from user: {name}\nThe information extracted from above command:",
    input_variables=["name"],
    example_separator="\n\n",
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt)

print(chain.run(name = "Nexian"))
# All prompts are loaded through the `load_prompt` function.
from langchain.prompts import load_prompt, PromptTemplate


template = "Tell me a {adjective} joke about {content}."
prompt = PromptTemplate(template=template, input_variables=["adjective", "content"])
# input_variables=['adjective', 'content'] template='Tell me a {adjective} joke about {content}.'

# save the prompt in JSON
prompt.save("simple_prompt.json")

# load the prompt from JSON file
prompt = load_prompt("simple_prompt.json")
print(prompt.format(adjective="funny", content="chickens"))
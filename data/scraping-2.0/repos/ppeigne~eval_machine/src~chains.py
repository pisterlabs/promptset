from config import config

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)






llm = ChatOpenAI(
    openai_api_key=config["openai_api_key"],
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate(
    template=config['main_prompt'],
    input_variables=["examples"]
)

prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate(prompt=prompt)])

chain = LLMChain(prompt=prompt,
                 llm=llm
)   

print(chain.run(examples="\n".join(config["examples"])))



# prompt = config["questions_prompt"]


# print(llm([HumanMessage(content="Hello, how are you?")]))

# print(chatbot(["Hello, how are you?"]))

# prompt_template = """
# {instruction}
# """
# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["instruction",
#                      "examples"]
# )

# 1. Generate one example from a fewshot prompt



# 1.1. Create a prompt template



# 1.2. Create a chat model

# 1.3. Create a chain




# 2. Evaluate the example correctness and  

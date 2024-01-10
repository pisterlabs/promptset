import chainlit as cl
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Replicate

# Initialize Replicate endpoint
# Alternative LLM options at https://replicate.com/explore
r8_llm =  Replicate(model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b")

# LangChain initialization
todo_prompt = """
Extract all potential to-dos or tasks from the following text in pure markdown. 
Do not make up information yourself, but you can summarize and suggest potential actions.
You can ask follow-up questions to get more context and better define the to-dos.

INPUT TEXT TO PROCESS:
{text}

For example:
I need to get in touch with Gert and Samantha about the open tickets and the new project.

- [ ] get in touch with Gert & Samantha about open tickets
- [ ] get in touch with Gert & Samantha about new project

What are the open tickets? And the new project? What are the actions you need Gert & Samantha to take?
"""

# Chainlit setup
@cl.langchain_factory(use_async=False)
def factory():
    todo_prompt_template = PromptTemplate(
        template=todo_prompt,
        input_variables=['text'],
    )
    todo_chain = LLMChain(
        llm=r8_llm,
        prompt=todo_prompt_template,
    )

    return todo_chain
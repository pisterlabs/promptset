from langchain import PromptTemplate

"""
By default, PromptTemplate will validate the template string by checking whether the input_variables match the variables defined in template. You can disable this behavior by setting validate_template to False
"""

template = "I am learning langchain because {reason}."

prompt_template = PromptTemplate(template=template, input_variables=["reason", "foo"]) # ValueError due to extra var

prompt_template = PromptTemplate(template=template, input_variables=["reason", "foo"], validate_template=False) # No error



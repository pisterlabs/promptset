from langchain import PromptTemplate


template = """
I want you to act as a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)
prompt.format(product="Mini Computers")
# -> I want you to act as a naming consultant for new companies.
# -> What is a good name for a company that makes Mini Computers?

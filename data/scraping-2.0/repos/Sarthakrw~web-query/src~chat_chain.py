from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain import PromptTemplate

llm = HuggingFacePipeline(pipeline=pipe)

system_message = """
You are a helpful, respectful and honest assistant. Your job is to answer the users query as best as possible given the Web Page Content. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. If you DO NOT KNOW THE ANSWER DO NOT SHARE FALSE INFORMATION.
You have been given scraped text content of a webpage under the section called "Web Page Content". Using this information answer the users query. However, if the webpage DOES NOT contain the answer to the query, you CAN answer based on your existing knowledge IF you are sure of the answer, but ALWAYS let the user know when doing so.
"""

prompt_template='''[INST] <<SYS>>
{system_message}
<</SYS>>

Web Page Content:
```
{context}
```

{prompt}[/INST]'''

chat = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
)

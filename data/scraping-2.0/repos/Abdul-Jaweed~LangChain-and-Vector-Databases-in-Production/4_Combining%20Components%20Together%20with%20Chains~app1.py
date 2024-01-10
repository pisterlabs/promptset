## Parsers

# As discussed, the output parsers can define a data schema to generate correctly formatted responses. It wouldn’t be an end-to-end pipeline without using parsers to extract information from the LLM textual output. The following example shows the use of CommaSeparatedListOutputParser class with the PromptTemplate to ensure the results will be in a list format.

from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")



output_parser = CommaSeparatedListOutputParser()

template = """List all possible words as substitute for 'artificial' as comma separated."""

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

prompt = PromptTemplate(
    template=template,
    output_parser=output_parser,
    input_variables=[]
)


llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=output_parser
)

llm_chain.predict()
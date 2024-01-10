import json
import os
import sys

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from bedrock.utils import bedrock, print_ww
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import XMLOutputParser
from langchain.schema.output_parser import StrOutputParser

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

'''
Split the long test into chunks
'''
shareholder_letter = "./data/2022-letter.txt"

with open(shareholder_letter, "r") as file:
    letter = file.read()
modelId = "anthropic.claude-v2"
llm = Bedrock(
    model_id=modelId,
    model_kwargs={
        'temperature': 0.3
    },
    client=boto3_bedrock,
)    
print(llm.get_num_tokens(letter))


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
)

docs = text_splitter.create_documents([letter])
num_docs = len(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

print(
    f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
)


xml_parser = XMLOutputParser(tags=['insight'])
str_parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
    
    Human:
    {instructions} : \"{document}\"
    Format help: {format_instructions}.
    Assistant:""",
    input_variables=["instructions","document"],
    partial_variables={"format_instructions": xml_parser.get_format_instructions()},
)

insight_chain = prompt | llm | StrOutputParser()

insights=[]
for i in range(len(docs)):
    insights.append(
        insight_chain.invoke({
        "instructions":"Provide Key insights from the following text",
        "document": {docs[i].page_content}
    }))

str_parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
    
    Human:
    {instructions} : \"{document}\"
    Assistant:""",
    input_variables=["instructions","document"]
)

summary_chain = prompt | llm | StrOutputParser()


print_ww(summary_chain.invoke({
        "instructions":"You will be provided with multiple sets of insights. Compile and summarize these insights and provide key takeaways in one concise paragraph. Do not use the original xml tags. Just provide a paragraph with your compiled insights.",
        "document": {'\n'.join(insights)}
    }))
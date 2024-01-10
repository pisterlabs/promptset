# StuffDocumentation https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.StuffDocumentsChain.html
# ReduceDocumentsChain https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.reduce.ReduceDocumentsChain.html
# MapReduceDocumentsChain https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.map_reduce.MapReduceDocumentsChain.html#langchain.chains.combine_documents.map_reduce.MapReduceDocumentsChain

"""Retriever and vector store"""
import asyncio
import time
from langchain.docstore.document import Document
import json
if True:
    import sys
    sys.path.append("../")
from src.utils.config import load_config
from src.base_llm import BaseLLM
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain, ReduceDocumentsChain, MapReduceDocumentsChain
import pickle
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator


from pprint import pprint

# Args
path_file = "../docs_example/contract.pdf"
debug = True
# Parameters
chunk_size = 1000
chunk_overlap = 0.1
document_variable_name = "context"


load_config(debug=debug)

# instance LLM
llm = BaseLLM(debug=debug)()

# Outputs strcuture
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# map output


class MapOutput(PydanticOutputParser):
    original_parragraph: list = Field(
        description="List of three original parragraphs")
    modified_parragraph: list = Field(
        description="List of three modified parragraphs")
    explanation: list = Field(
        description="List of three explanations of why the parragraph should be modified")


def parse_output_suggestions(response_str: str) -> list:
    """Parse output suggestions
    Args:
        response: response from the chain
    Returns:
        List of tuples with the original parragraph and the modified parragraph
        example:
        [
            ("original parragraph 1", "modified parragraph 1"),
            ("original parragraph 2", "modified parragraph 2"),
            ("original parragraph 3", "modified parragraph 3"),
        ]

    """
    response_dict = json.loads(response_str)

    original_parragraph_ls = []
    for r in response_dict['original_parragraph']:
        if type(r) == str:
            original_parragraph_ls.append(r)
        if type(r) == dict:
            if 'Original parragraph' in list(r.keys()):
                original_parragraph_ls.append(r['Original parragraph'])
            if 'Original Parragraph' in list(r.keys()):
                original_parragraph_ls.append(r['Original Parragraph'])

    modified_parragraph_ls = []
    for r in response_dict['modified_parragraph']:
        if type(r) == str:
            modified_parragraph_ls.append(r)
        if type(r) == dict:
            if 'Modified parragraph' in list(r.keys()):
                modified_parragraph_ls.append(r['Modified parragraph'])
            if 'Modified Parragraph' in list(r.keys()):
                modified_parragraph_ls.append(r['Modified Parragraph'])

    explanation_ls = []
    for r in response_dict['explanation']:
        if type(r) == str:
            explanation_ls.append(r)
        if type(r) == dict:
            if 'Explanation' in list(r.keys()):
                explanation_ls.append(r['Explanation'])
            if 'explanation' in list(r.keys()):
                explanation_ls.append(r['explanation'])

    # join the original parragraphs and modified parragraphs as tuple
    res_parsed = {}
    res_parsed['suggestions'] = []
    for i in range(len(original_parragraph_ls)):
        res_parsed['suggestions'].append({
            'orgiginal_parragraph': original_parragraph_ls[i],
            'modified_parragraph': modified_parragraph_ls[i],
            'explanation': explanation_ls[i],
        })

    # res_parsed = list(zip(original_parragraph_ls,
        #   modified_parragraph_ls, explanation_ls))
    return res_parsed


parser_reduce = PydanticOutputParser(pydantic_object=MapOutput)


# Prompts
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
PROMPT_DOC = PromptTemplate(
    template="{page_content}",
    input_variables=["page_content"]
)

prompt_map_temp = """Use the following pieces to generate one suggestion to modify clauses in the contract provided below that may potentially harm the client's interests. Highlight potential risks and propose alternative language or conditions to mitigate these risks.

{context}

The output should be formatted as a the schema below:

Original parragraph: this is the original parragraph that we want to modify
Modified parragraph: this is the modified parragraph
Explanation: this is the explanation of why the parragraph should be modified

Do this for the top 3 suggestions.
Do not add any other text to the output.
Do not add numbers or letters or make any modification to the keys "Original parragraph", "Modified parragraph" and "Explanation".

"""

prompt_map = PromptTemplate(
    template=prompt_map_temp,
    input_variables=["context"],
    # partial_variables={
    #     "format_instructions": parser_map.get_format_instructions()},
    # output_parser=parser_map
)

prompt_reduce_temp = """Select from the following suggestions the best three suggestions.

{context}

The output should be formatted as a the schema below:

Original parragraph: this is the original parragraph that we want to modify
Modified parragraph: this is the modified parragraph
Explanation: this is the explanation of why the parragraph should be modified

Do this for the top 3 suggestions.
Do not add any other text to the output.
Do not add numbers or letters or make any modification to the keys "Original parragraph", "Modified parragraph" and "Explanation".

Response:
"""
prompt_reduce = PromptTemplate(
    template=prompt_reduce_temp,
    input_variables=["context"],
    # partial_variables={
    #    "format_instructions": parser_reduce.get_format_instructions()},
)


# Chains
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# create map chain
map_chain = LLMChain(
    llm=llm,
    prompt=prompt_map,
    verbose=debug,
)

# create reduce chain
reduce_chain = LLMChain(
    llm=llm,
    prompt=prompt_reduce,
    verbose=debug
)

# transform list of docs to string
docs_to_string_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,
    document_prompt=PROMPT_DOC,
    document_variable_name=document_variable_name)

# Combine docs by recursivelly reducing them
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=docs_to_string_chain,
)

# map reduce chain
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
)


# Spliiter
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# load doc
doc = PDFMinerLoader(path_file).load()
# split chunks
spliiter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = spliiter.split_documents(doc)[:10]
# pprint(chunks[3].to_json()['kwargs']['page_content'])

# Run Map redcue chain
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
astart_time = time.time()

print(map_chain.run(chunks))


async def run():
    res = await asyncio.gather(map_reduce_chain.arun(chunks))
    return res
res = run()
if type(res) == list:
    res = res[0]
print(res)

aend_time = time.time()

start_time = time.time()
final_res = map_reduce_chain.run(chunks)
print(final_res)
end_time = time.time()
print("--- %s sequential seconds ---" %
      (end_time - start_time))      # 1-31.8-openai | 4-50.2-openai | 10-170-openai seconds
print("--- %s async seconds --------" %
      (aend_time - astart_time))    # 1-22.8-openai | 4-29.1-openai | 10-54.6-openai seconds | 10-9.2-bedrock seconds

# format output
final_res_parsed = parse_output_suggestions(res[0])
# show output
pprint(final_res_parsed)


final_res_parsed['suggestions'][0]
"""
SEQUENTIAL --- 64.99065089225769 seconds ---
"""

print("--- %s seconds ---" % (end_time))


import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


import re
# # Embedding Support
# from langchain.embeddings import HuggingFaceEmbeddings

# Summarizer using Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Taking out the warnings
import warnings
from warnings import simplefilter
import time

## Get Relevance of papers to RQ ##
response_schemas = [
    ResponseSchema(name="answer", description="Helpful answer on relavance to reserch question, with explanation"),
    ResponseSchema(name="score", description="Score between 0 and 100"),
    ResponseSchema(name="Title", description="Title of the paper"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt_template = """Use the following Titles and abstracts to asses if each is relevant to the research question(s). If so, provide a helpful answer as to why. 
If not, provide a helpful answer as to why not. Provide a score between 0 and 100, where 100 is most relevant and 0 is not relevant at all.
\n{format_instructions}
Begin!

Titles and Abstracts:
---------
{context}
---------
Research question(s): {question}
"""


def get_relavance_abstracts(docs, RQ, openai_api_key):

    model = OpenAI(temperature=0, openai_api_key=openai_api_key)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions}
    )

    if len(RQ) == 1:
        RQ = RQ[0]
    elif len(RQ) > 1:
        RQ = RQ[0] + ' and ' + RQ[1]

    relevance = {}
    for doc in docs:
        _input = prompt.format_prompt(context=doc, question=RQ)
        result = model(_input.to_string())
        output = output_parser.parse(result)
        relevance[output['Title']] = {'score' : output['score'], 'answer' : output['answer']}
    sorted_docs = sorted(relevance.items(), key=lambda x: int(x[1]['score']), reverse=True)
    return sorted_docs

# Filter out FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

## STEP 1: create an agent that will give a summary of an article in relation to the users research question(s) ##
map_prompt = """
You will be given a single passage of an article. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a concise understanding of the articles: purpose, findings or method (which ever applies to the passage).
Whenever you revice a methods section, you should include extra details of the methods used in the article.
Specifically in relation to the readers Research question(s): {}.
"""

map_reduce = """
Your response should be at least one paragraphs and fully encompass what was said in the passage.

```{text}```
CONCISE SUMMARY:
"""

combine_prompt = """
You will be given a series of summaries from an article. The summaries will be enclosed in triple backticks (```)
Your goal is to: 1. give a verbose BULLET POINT summary in relation to the readers Research question(s): {}. 
2. simplify the language by use of synonyms and paraphrasing if the language is too technical.
Return your response in bullet points which covers how the article relates their research question and the articles purpose, findings and methods, in that order and within a maximum of 10 points.
Finish by a conclusion point which states whether the article is useful in relation to the readers research question. subtitle the bullet points with ## purpose,  ## methods, ## findings, ## conclusion.
"""
combine_reduce = """
```{text}```
BULLET POINT SUMMARY:
"""
st.cache_resource(ttl=60*30)
def generate_response(docs, openai_api_key, RQ):
    chunk_size = 2000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    # get the start time
    st = time.time()
    # Instantiate the LLM model
    map_prompt_reduced = map_prompt.format(RQ) + map_reduce

    map_prompt_template = PromptTemplate(template=map_prompt_reduced, input_variables=["text"])
    print(map_prompt_template)

    combine_prompt_reduced = combine_prompt.format(RQ) + combine_reduce
    print(combine_prompt_reduced)
    combine_prompt_template = PromptTemplate(template=combine_prompt_reduced, input_variables=["text"])
    llm = ChatOpenAI(temperature=0,
                    openai_api_key=openai_api_key,
                    max_tokens=600,
                    model='gpt-3.5-turbo'
                    )
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce',
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template,
                                 verbose=True)


    response = chain.run(doc)
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f"Elapsed time Chain: {elapsed_time} seconds")
    return response

def clean_pdf(pages):
    pattern = re.compile(r'(?m)(\w+\,\s\w\.)(.?\;?)*\s\(?\d+\)?(.*?)\(.*\)?')
    clean_pages = [] 
    # count if pattern is found > 2 times in a page
    for page in pages[:-1]:
        if len(re.findall(pattern, page)) <= 1:
            clean_pages.append(page)
    return clean_pages



### STEP 2: create and agent that will give a litirary review ### ~ and a ranked? list of articles based on the users research question, commenting on the articles relevance to the research question.##

lit_prompt = """You are a helpful AI bot that aids a user in research. You will be given a research question and a list of summaries from articles. 
Your goal is to give a literary review of the articles in relation to the research question. Specifically, you will give a summary of the articles purpose, findings and especially the method, and refer to them continously.
you will be given the title of the article and a bullet point list about the article inside triple backticks (```). Keep the summary within two paragraphs.
The Research question(s) are: {}
"""
lit_reduce = """
```{text}```
LITERARY REVIEW:
"""
@st.cache_resource(ttl=60*30)
def lit_review(file_dict, RQ, openai_api_key):
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model='gpt-3.5-turbo')
    if len(RQ) == 1:
            RQ = RQ[0]
    elif len(RQ) > 1:
        RQ = RQ[0] + ' and ' + RQ[1]

    lit_prompt_reduced = lit_prompt.format(RQ) + lit_reduce
    print(lit_prompt_reduced)
    lit_prompt_template = PromptTemplate(template=lit_prompt_reduced, input_variables=["text"])
    # Initialize variables
    max_tokens = 1200  # Set your desired token limit here
    summary = []
    current_summary = ""

    # Loop through the items in the dictionary
    for key, item in file_dict.items():
        # Check if adding the current item would exceed the token limit
        if (len(current_summary) + len(item.split())) // 4 <= max_tokens:
            current_summary += 'title: ' + key +'\n' + item + " "
        else:
            # If adding the current item would exceed the limit, append the current summary to the list
            summary.append(current_summary.strip())
            # Start a new summary with the current item
            current_summary = item + " "

    # Append the last summary if it's not empty
    if current_summary:
        summary.append(current_summary.strip())

    # Print the list of joined items
    print(len(summary))
    print(summary)
    lit_chain = load_summarize_chain(llm=llm,
                                    chain_type="stuff",
                                    prompt=lit_prompt_template)
    file_dict['lit_review'] = []
    for i, s in enumerate(summary):
        docs = [Document(page_content=s)]
        print(docs)
        final_summary = lit_chain.run(docs)
        file_dict['lit_review'].append(final_summary)
    return file_dict
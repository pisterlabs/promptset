

# imports
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from langchain import PromptTemplate,  LLMChain
import os 
import pymysql as mdb
from time import gmtime, strftime
import time



# constants
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_LLM = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in sentences which covers the key points of the text.
```{input_text}```
BULLET POINT SUMMARY:
"""

PROMPT_COMMAND = "Summarize the information related to {} from the following text delimited by triple backquotes.\n"
PROMPT_LLM_GENE = """
```{input_text}```
SUMMARY:
"""


# DB constants
DB_PASSWD = os.environ.get('DB_PASSWD')
NUM_ABSTRACT_LIMIT = 5
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

SQL_SELECT_ABSTRACT_BY_TITLE = "select id from {}.pgpt_paper_abstract where title = %s".format(SCHEMA_GPT)
SQL_SELECT_ABSTRACT_LIST_LEVEL_0 = """
select abst.id, abst.abstract 
from {}.pgpt_paper_abstract abst, {}.pgpt_search_paper seapaper 
where abst.document_level = 0 and seapaper.paper_id = abst.pubmed_id and seapaper.search_id = %s limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)
# and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s

SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER = """
select distinct abst.id, abst.abstract, abst.document_level
from {}.pgpt_paper_abstract abst, {}.pgpt_gpt_paper gpt
where abst.document_level = %s and gpt.parent_id = abst.id and gpt.search_id = %s
and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)

# SQL_INSERT_PAPER = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (abstract, title, journal_name, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_INSERT_GPT_LINK = "insert into {}.pgpt_gpt_paper (search_id, parent_id, child_id, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_UPDATE_ABSTRACT_FOR_TOP_LEVEL = "update {}.pgpt_paper_abstract set search_top_level_of = %s where id = %s".format(SCHEMA_GPT)

SQL_SELECT_SEARCHES = "select id, terms, gene from {}.pgpt_search where ready='Y' limit %s".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_AFTER_SUMMARY = "update {}.pgpt_search set ready = 'N', date_last_summary = sysdate() where id = %s ".format(SCHEMA_GPT)

# methods
def get_model_tokenizer(name, log=False):
    '''
    returns the associated model and tokenizer
    '''
    # initialize
    tokenizer = AutoTokenizer.from_pretrained(name)
    pipeline = transformers.pipeline(
        "text-generation", #task
        model=name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    model = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    # return
    return model, tokenizer


def summarize(model, gene, command, prompt, text_to_summarize, log=False):
    '''
    summarixes the test input using the model and prompt provided
    '''
    # initialize
    command_gene = command.format(gene)
    prompt_and_command = command_gene + prompt

    if log:
        print("prompt: \n{}".format(prompt_and_command))
    prompt_final = PromptTemplate(template=prompt_and_command, input_variables=["input_text"])
    llm_chain = LLMChain(prompt=prompt_final, llm=model)
    summary = None

    # log
    if log:
        print("using prompt: \n{}".format(prompt_and_command))

    # summarize
    summary = llm_chain.run(text_to_summarize)

    # return
    return summary

def get_list_abstracts(conn, id_search, num_level=0, num_abstracts=NUM_ABSTRACT_LIMIT, log=False):
    '''
    get a list of abstract map objects
    '''
    # initialize
    list_abstracts = []
    cursor = conn.cursor()

    # pick the sql based on level
    if log:
        print("searching for abstracts got input search: {}, doc_level: {}, limit: {}".format(id_search, num_level, num_abstracts))
    if num_level == 0:
        # cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, id_search, num_abstracts))
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, num_abstracts))
    else:
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER, (num_level, id_search, id_search, num_abstracts))

    # query 
    db_result = cursor.fetchall()
    for row in db_result:
        paper_id = row[0]
        abstract = row[1]
        list_abstracts.append({"id": paper_id, 'abstract': abstract})

    # return
    return list_abstracts

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # # initialize
    # num_level = 0
    # id_search = 2

    # # get the db connection
    # conn = get_connection()

    # # get the abstracts
    # list_abstracts = get_list_abstracts(conn=conn, id_search=id_search, num_level=num_level, num_abstracts=5, log=True)

    # # get the llm summary
    # str_input = ""
    # if len(list_abstracts) > 1:
    #     # top level is not this level if more than 2 abstracts found at this level
    #     found_top_level = False
    #     for item in list_abstracts:
    #         abstract = item.get('abstract')
    #         print("using abstract: \n{}".format(abstract))
    #         str_input = str_input + " " + abstract

    #     # log
    #     print("using {} for gpt query for level: {} and search: {}".format(len(list_abstracts), num_level, id_search))

    # # get the model
    # llm_model, tokenizer = get_model_tokenizer(MODEL_NAME)
    # print("got model: {}".format(llm_model))

    # # get the summary
    # summary = summarize(model=llm_model, gene='UBE2NL', command=PROMPT_COMMAND, prompt=PROMPT_LLM_GENE, text_to_summarize=str_input, log=True)
    # print("got summary: \n{}".format(summary))


    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    template = """
                Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
            """

    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    text = """ As part of Meta’s commitment to open science, today we are publicly releasing LLaMA (Large Language Model Meta AI), a state-of-the-art foundational large language model designed to help researchers advance their work in this subfield of AI. Smaller, more performant models such as LLaMA enable others in the research community who don’t have access to large amounts of infrastructure to study these models, further democratizing access in this important, fast-changing field.

    Training smaller foundation models like LLaMA is desirable in the large language model space because it requires far less computing power and resources to test new approaches, validate others’ work, and explore new use cases. Foundation models train on a large set of unlabeled data, which makes them ideal for fine-tuning for a variety of tasks. We are making LLaMA available at several sizes (7B, 13B, 33B, and 65B parameters) and also sharing a LLaMA model card that details how we built the model in keeping with our approach to Responsible AI practices.

    Over the last year, large language models — natural language processing (NLP) systems with billions of parameters — have shown new capabilities to generate creative text, solve mathematical theorems, predict protein structures, answer reading comprehension questions, and more. They are one of the clearest cases of the substantial potential benefits AI can offer at scale to billions of people.

    Even with all the recent advancements in large language models, full research access to them remains limited because of the resources that are required to train and run such large models. This restricted access has limited researchers’ ability to understand how and why these large language models work, hindering progress on efforts to improve their robustness and mitigate known issues, such as bias, toxicity, and the potential for generating misinformation.

    Smaller models trained on more tokens — which are pieces of words — are easier to retrain and fine-tune for specific potential product use cases. We trained LLaMA 65B and LLaMA 33B on 1.4 trillion tokens. Our smallest model, LLaMA 7B, is trained on one trillion tokens.

    Like other large language models, LLaMA works by taking a sequence of words as an input and predicts a next word to recursively generate text. To train our model, we chose text from the 20 languages with the most speakers, focusing on those with Latin and Cyrillic alphabets.

    There is still more research that needs to be done to address the risks of bias, toxic comments, and hallucinations in large language models. Like other models, LLaMA shares these challenges. As a foundation model, LLaMA is designed to be versatile and can be applied to many different use cases, versus a fine-tuned model that is designed for a specific task. By sharing the code for LLaMA, other researchers can more easily test new approaches to limiting or eliminating these problems in large language models. We also provide in the paper a set of evaluations on benchmarks evaluating model biases and toxicity to show the model’s limitations and to support further research in this crucial area.

    To maintain integrity and prevent misuse, we are releasing our model under a noncommercial license focused on research use cases. Access to the model will be granted on a case-by-case basis to academic researchers; those affiliated with organizations in government, civil society, and academia; and industry research laboratories around the world. People interested in applying for access can find the link to the application in our research paper.

    We believe that the entire AI community — academic researchers, civil society, policymakers, and industry — must work together to develop clear guidelines around responsible AI in general and responsible large language models in particular. We look forward to seeing what the community can learn — and eventually build — using LLaMA.
    """


    print(llm_chain.run(text))




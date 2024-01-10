"""Drops a collection from the document storage."""
import os

import json
from pathlib import Path
import pprint
import pdb
from typing import Any

from etl import markdown, pdfs, shared, videos
from etl.shared import display_modal_image

import docstore
import vecstore
from utils import pretty_log

pp = pprint.PrettyPrinter(indent=2)

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import json
import textwrap
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import time


def log_event(query: str, sources, answer: str, request_id=None):
    """Logs the event to Gantry."""
    import os

    import gantry

    if not os.environ.get("GANTRY_API_KEY"):
        pretty_log("No Gantry API key found, skipping logging")
        return None

    gantry.init(api_key=os.environ["GANTRY_API_KEY"], environment="modal")

    application = "dynabicChatbot"
    join_key = str(request_id) if request_id else None

    inputs = {"question": query}
    inputs["docs"] = "\n\n---\n\n".join(source.page_content for source in sources)
    inputs["sources"] = "\n\n---\n\n".join(
        source.metadata["source"] for source in sources
    )
    outputs = {"answer_text": answer}

    record_key = gantry.log_record(
        application=application, inputs=inputs, outputs=outputs, join_key=join_key
    )

    return record_key


def prep_documents_for_vector_storage(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100, allowed_special="all"
    )
    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    return ids, texts, metadatas



# TODO: use 8 CPUs here use parallelism !!!
def create_vector_index(collection: str = None, db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    pretty_log(f"collecting documents from {collection.name}")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs)

    pretty_log(f"sending to vector index {vecstore.INDEX_NAME}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(
        vecstore.INDEX_NAME, embedding_engine, texts, metadatas
    )
    vector_index.save_local(folder_path=vecstore.VECTOR_DIR, index_name=vecstore.INDEX_NAME)
    pretty_log(f"vector index {vecstore.INDEX_NAME} created")

def qanda(query: str, request_id=None, with_logging: bool = False) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.chat_models import ChatOpenAI
    import prompts
    import vecstore

    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256) # gpt4 no longer available for free
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts.main,
        document_variable_name="sources",
    )

    result = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]

    if with_logging:
        print(answer)
        pretty_log("logging results to gantry")
        record_key = log_event(query, sources, answer, request_id=request_id)
        if record_key:
            pretty_log(f"logged to gantry with key {record_key}")

    return answer

def transform_papers_to_json():
    papers_path = Path("data") / "pdfpapers.json"

    with open(papers_path) as f:
        pdf_infos = json.load(f)

    # print(pdf_infos[:100:20])

    # E nrich the paper data by finding direct PDF URLs where we can
    paper_data = map(pdfs.get_pdf_url, pdf_infos[::25])

    # turn the PDFs into JSON documents
    it = map(pdfs.extract_pdf, paper_data)
    documents = shared.unchunk(it)

    # Store the collection of docs on the server
    docstore.drop(os.environ["MONGODB_COLLECTION"], os.environ["MONGODB_DATABASE"])

    # Test out debug
    #pp.pprint(documents[0]["metadata"])

    # Split document list into 10 pieces
    chunked_documents = shared.chunk_into(documents, 10)
    results = list(map(shared.add_to_document_db, chunked_documents))

    # Pull only arxiv other_papers
    query = {"metadata.source": {"$regex": "arxiv\.org", "$options": "i"}}
    # Project out the text field, it can get large
    projection = {"text": 0}
    # get just one result to show it worked
    result = docstore.query_one(query, projection)


    pp.pprint(result)

def solve_vector_storage():
    VECTOR_DIR = vecstore.VECTOR_DIR
    vector_storage = "vector-vol"

    create_vector_index(os.environ["MONGODB_COLLECTION"], os.environ["MONGODB_DATABASE"])



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

tokenizer = None
model = None
embedding_engine = None
base_llm = None

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text


llm_chain = None

def qanda_llama2(query: str, request_id=None, with_logging: bool = False) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    import prompts
    import vecstore

    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    # MODEL LOADING STUFF

    global tokenizer
    global model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                              use_auth_token=True, )

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 token=True,
                                                 #  load_in_8bit=True,
                                                 #  load_in_4bit=True
                                                 )

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto", # device = 0?
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.1,
                    top_k=0.95,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id = tokenizer.eos_token_id,
                    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

    instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
    system_prompt = "You are an interviewer for deciding whether to hire a person as a security officer. Please ask technical question about various things in the field"
    template = get_prompt(instruction, system_prompt)
    print(template)

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    global llm_chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    return

    ################# PART 2 ###################
    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")


    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256) # gpt4 no longer available for free
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts.main,
        document_variable_name="sources",
    )

    result = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]

    if with_logging:
        print(answer)
        pretty_log("logging results to gantry")
        record_key = log_event(query, sources, answer, request_id=request_id)
        if record_key:
            pretty_log(f"logged to gantry with key {record_key}")

    return answer

def ask_question(query):
    result = llm_chain.predict(user_input=query); #({"question": query}, return_only_outputs=True)
    #answer = result["output_text"]
    return result

def qanda_llama2_withRAG(query: str, request_id=None, with_logging: bool = False) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    import prompts
    import vecstore

    global embedding_engine
    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    # MODEL LOADING STUFF

    global tokenizer
    global model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                              use_auth_token=True, )

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 use_auth_token=True,
                                                 #  load_in_8bit=True,
                                                 #  load_in_4bit=True
                                                 )

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=512,
                    do_sample=True,
                    top_k=1,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

    instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
    system_prompt = "You are an interviewer for deciding whether to hire a person as a security officer. Please ask technical question about various things in the field"
    template = get_prompt(instruction, system_prompt)
    print(template)

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    global llm_chain
    llm_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts.main,
        document_variable_name="sources",
    )

def ask_question(query):
    result = llm_chain.predict(user_input=query); #({"question": query}, return_only_outputs=True)
    #answer = result["output_text"]
    return result

def ask_question_withRAG(query, with_logging=True):
    ################# PART 2 ###################
    
    start_con = time.time()
    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    
    start_sim_search = time.time()

    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    #pretty_log("running query against Q&A chain", sources, scores)

    start_llm_ask = time.time()

    result = llm_chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]
    
    end_llm_ask = time.time()

    if with_logging:
        print(answer)
        pretty_log("logging results to gantry")
        record_key = log_event(query, sources, answer, request_id=request_id)
        if record_key:
            pretty_log(f"logged to gantry with key {record_key}")

    print(answer)
    
    print(f"Total time: {end_llm_ask-start_con}\n Conn to vector store: {start_sim_search-start_con}\n\
        Sim search: {start_llm_ask-start_sim_search}\n LLm ask: {end_llm_ask-start_llm_ask}\n")


from langchain.callbacks.base import BaseCallbackHandler
class MyCustomStreamingCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # do your things instead of stdout
        print(token)


def ask_question_llama2_cont(query):
    #result = llm_chain.predict(user_input=query); #({"question": query}, return_only_outputs=True)
    #answer = result["output_text"]

    result1 = llm_chain({"user_input" : "Give me some indications to solve a denial of service attack"})
    print(result1)
    result2 = llm_chain({"user_input": "What did I ask you previously?"})
    print(result2)

    return result2



def qanda_llama2_cont(request_id=None, with_logging: bool = False) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    import prompts
    import vecstore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline


    global embedding_engine
    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    # MODEL LOADING STUFF

    global tokenizer
    global model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                              token=True)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.bfloat16,
                                                 token=True,
                                                 #  load_in_8bit=True,
                                                 #  load_in_4bit=True,
                                                 )
    # use class TextIteratorStreamer(TextStreamer): check for gradio example C:\Users\cipri\AppData\Local\JetBrains\PyCharm2023.3\remote_sources\-643487973\-1298038738\transformers\generation\streamers.py
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=4096,
                    do_sample=True,
                    #temperature=0.1,
                    top_p=0.95,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id = tokenizer.eos_token_id,
                    streamer=streamer,
                    )

    llm = HuggingFacePipeline(pipeline=pipe)
    
    global base_llm
    base_llm = llm

    instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
    system_prompt = """\
        ""Consider that I'm a beginner in networking and security things. \n
        Give me a concise answer with with a single step at a time. \n
        Limit your resonse to maximum 128 words.
        Do not provide any additional text or presentation. Only steps and actions.
        If possible use concrete names of software or tools that could help on each step."""
    template = get_prompt(instruction, system_prompt)
    print(template)

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    global llm_chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    #device=0
    #inputs = tokenizer("Give me some ideas", return_tensors="pt").to(device)
    #streamer = TextStreamer(tokenizer, skip_prompt=True)
    #llm_chain({"user_input": "Give me some indications to solve a denial of service attack"})

    return

    ################# PART 2 ###################
    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")


    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256) # gpt4 no longer available for free
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts.main,
        document_variable_name="sources",
    )

    result = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]

    if with_logging:
        print(answer)
        pretty_log("logging results to gantry")
        record_key = log_event(query, sources, answer, request_id=request_id)
        if record_key:
            pretty_log(f"logged to gantry with key {record_key}")

    return answer

def custom_test():
    import bs4
    from langchain import hub
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import WebBaseLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema import StrOutputParser
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable.utils import ConfigurableField
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_engine)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    prompt = hub.pull("rlm/rag-prompt-llama")
    llm = base_llm

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )
    chat_history = []
    query = "What did the president say about Ketanji Brown Jackson"
    result = chain({"question": query, "chat_history": chat_history})



    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    ###########
    from langchain.prompts import PromptTemplate
    from operator import itemgetter
    from langchain.schema.runnable import RunnableParallel

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use 4 sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    rag_prompt_custom = ChatPromptTemplate.from_template(template)

    # Always say "thanks for asking!" at the end of the answer.

    rag_chain_from_docs = (
            {
                "context": lambda input: format_docs(input["documents"]),
                "question": itemgetter("question"),
            }
            | rag_prompt_custom
            | llm
            | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
                                "documents": lambda input: [doc.metadata for doc in input["documents"]],
                                "answer": rag_chain_from_docs,
                            }

    #rag_chain_with_source.invoke({'question': "What is Task Decomposition?"})
    #rag_chain_with_source.invoke("What is Task Decomposition?")

    rag_chain({'question': "What is Task Decomposition?"})

def another_common_test():
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import pipeline, TextStreamer
    import json
    import textwrap
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory

    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                              token=True)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.bfloat16,
                                                 token=True,
                                                 #  load_in_8bit=True,
                                                 #  load_in_4bit=True,
                                                 )
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=4096,
                    do_sample=True,
                    # temperature=0.1,
                    top_p=0.95,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    )

    llm = HuggingFacePipeline(pipeline=pipe)

    from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
    from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(vecstore.INDEX_NAME, embedding_engine)
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    doc_chain_custom_prompt="""Given the following extracted parts of a long document and a question, create a final answer with sources ("SOURCES") that represent parts of the given summaries
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}

{summaries}

FINAL ANSWER:"""

    qa_prompt = PromptTemplate(
        template=doc_chain_custom_prompt, input_variables=["context", "question"]
    )

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt= qa_prompt, verbose=True)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain(
        retriever=vector_index.as_retriever(search_kwargs={'k': 6}),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        memory=memory
    )

    chat_history = []
    query = "What models use human instructions?"
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)
    print(sources_and_scores)

    # doc_chain({"question":"What models use human instructions?", "input_documents":sources})

    chain({"question": "What models use human instructions?"})

    chain({"question": "Which are the advantage of each of these models?"})

    chain({"question": "Can you elaborate more on point 3?"})
def __main__():
    another_common_test()

    #response = qanda_llama2("Can we combine LMMs and OCR?", with_logging=True)
    #print(response)

    #qanda_llama2_cont()
    #res = ask_question_llama2_cont("Give me some indications to solve a phishing attack")
    #custom_test()
    #print(res)

if __name__ == "__main__":
    __main__()

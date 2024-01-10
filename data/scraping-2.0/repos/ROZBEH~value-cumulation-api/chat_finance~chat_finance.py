import datetime
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# langchain imports
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAIChat
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from sec_api import QueryApi, RenderApi

# from logger import custom_logger
from llama_index import (
    GPTVectorStoreIndex,
    ListIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    download_loader,
    load_graph_from_storage,
    load_index_from_storage,
)
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaToolkit,
    create_llama_chat_agent,
)
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
SEC_API_KEY = os.getenv("SEC_API_KEY")
SEC_API_ENDPOINT = os.getenv("SEC_API_ENDPOINT")
queryApi = QueryApi(api_key=SEC_API_KEY)
renderApi = RenderApi(api_key=SEC_API_KEY)


def index_txt_doc(path) -> GPTVectorStoreIndex:
    """
    Reads the contents of a text file and creates a GPTVectorStoreIndex from it.

    Args:
        file_path (str): The path to the text file to read.

    Returns:
        index (GPTVectorStoreIndex): The GPTVectorStoreIndex created from the text file.

    """
    documents = SimpleDirectoryReader(path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    return index


def index_html_doc(path) -> GPTVectorStoreIndex:
    """
    Reads the contents of an html file and creates a GPTVectorStoreIndex from it.

    Args:
        file_path (str): The path to the html file to read.

    Returns:
        index (GPTVectorStoreIndex): The GPTVectorStoreIndex created from the text file.

    """
    if not path.endswith(".html") and not path.endswith(".htm"):
        raise ValueError("The file must end in .html or .htm")
    llm_predictor = LLMPredictor(
        # llm=ChatOpenAI(model_name="gpt-4", max_tokens=512, temperature=0)
        llm=OpenAIChat(model_name="gpt-4", max_tokens=512, temperature=0)
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=512
    )
    if not Path("./storage").is_dir():
        print("creating index")
        UnstructuredReader = download_loader("UnstructuredReader")
        loader = UnstructuredReader()
        document = loader.load_data(file=Path(path), split_documents=False)
        index = GPTVectorStoreIndex.from_documents(
            document, service_context=service_context
        )
        index.storage_context.persist()
    else:
        print("loading index")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # load index
        index = load_index_from_storage(storage_context)

    return index


def index_sec_url(report_type="10-K", ticker="AAPL", year=None) -> GPTVectorStoreIndex:
    """
    Reads the contents of an html file and creates a GPTVectorStoreIndex from it.

    Args:
        report_type (str): Type of report to download
        ticker (str): Ticker of the company to download

    Returns:
        index (GPTVectorStoreIndex): The GPTVectorStoreIndex created from the text file.

    """
    html_file = None
    if not is_ticker_in_file(ticker):
        html_file = fetch_sec_report(report_type, ticker, year)

    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=512, temperature=0)
        # llm=OpenAIChat(model_name="gpt-4", max_tokens=512, temperature=0)
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=512
    )

    storage_path = os.path.join(current_dir, "storage")
    if html_file:
        if Path(storage_path).is_dir():
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            # custom_logger.info("Index found. Loading ...")
            # custom_logger.info(f"Adding {ticker} to the index")
            UnstructuredReader = download_loader("UnstructuredReader")
            loader = UnstructuredReader()
            document = loader.load_data(file=Path(html_file), split_documents=False)
            # load index
            index = load_index_from_storage(
                storage_context, service_context=service_context
            )

            index.insert(document[0])
            index.storage_context.persist(persist_dir=storage_path)
        else:
            # custom_logger.info("No index found. Creating ...")
            # reading the json html file approach
            UnstructuredReader = download_loader("UnstructuredReader")
            loader = UnstructuredReader()
            document = loader.load_data(file=Path(html_file), split_documents=False)
            # read the txt file approach
            index = GPTVectorStoreIndex.from_documents(
                document,
                service_context=service_context,
            )
            index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        # custom_logger.info(f"{ticker} is already in the index. Loading ...")
        UnstructuredReader = download_loader("UnstructuredReader")
        index = load_index_from_storage(
            storage_context, service_context=service_context
        )

    return index


def download_sec_urls(all_reports_df, ticker="AAPL", report_type="10-K", num_years=5):
    year = datetime.datetime.now().year

    query = {
        "query": {
            "query_string": {
                "query": f'ticker:{ticker} AND filedAt:{{{year-num_years}-11-30 TO {year}-11-30}} AND formType:"{report_type}" AND NOT formType:"10-K/A" AND NOT formType:NT'
            }
        },
        "from": "0",
        # "size": "10", # 10 of these documents
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    response = queryApi.get_filings(query)
    substring = "/ix?doc="
    if response:
        filings = response["filings"]
    else:
        raise ValueError("No filings found")
    # rename the entries
    filings = list(
        map(
            lambda f: {
                "ticker": f["ticker"],
                "cik": f["cik"],
                "formType": f["formType"],
                "filedAt": f["filedAt"],
                "filingUrl": f["linkToFilingDetails"],
                "inIndex": False,
            },
            filings,
        )
    )

    for filing in filings:
        sec_url = filing["filingUrl"]
        if isinstance(sec_url, str) and substring in sec_url:
            sec_url = sec_url.replace(substring, "")
        filing["filingUrl"] = sec_url
    new_reports_df = pd.DataFrame.from_records(filings)
    # if all_reports_df is not empty
    if not all_reports_df.empty:
        new_reports_df = new_reports_df[
            ~new_reports_df["filingUrl"].isin(all_reports_df["filingUrl"])
        ]

    all_reports_df = pd.concat([all_reports_df, new_reports_df])

    return new_reports_df, all_reports_df


def download_sec_report(sec_url, as_pdf=True):
    if as_pdf:
        api_url = (
            SEC_API_ENDPOINT + "?token=" + SEC_API_KEY + "&url=" + sec_url + "&type=pdf"
        )
        filing = requests.get(api_url)
    else:
        filing = renderApi.get_filing(sec_url)

    return filing


def prepare_sec_documents(
    company,
    all_reports_df=pd.DataFrame(),
    report_type="10-K",
    num_years=5,
    as_pdf=True,
):
    """
    Downloads and saves SEC filings for a given company and report type.

    Args:
        company (str): The ticker symbol of the company to download filings for.
        all_reports_df (pandas.DataFrame, optional): A DataFrame containing previously downloaded filings. Defaults to an empty DataFrame.
        report_type (str, optional): The type of report to download. Defaults to "10-K".
        year (int, optional): The year to start downloading filings from. If None, defaults to the second most recent year. Defaults to None.
        num_years (int, optional): The number of years of filings to download. Defaults to 5.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing two DataFrames. The first DataFrame contains the newly downloaded filings, and the second DataFrame contains all downloaded filings (old and new).
    """
    data_path = os.path.join(current_dir, "storage", "data", "sec10K")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # new_reports will be reports that are going to be downloaded
    # all_reports is old reports + new_reports

    company_path = os.path.join(data_path, company)
    new_reports_df, all_reports_df = download_sec_urls(
        all_reports_df, company, report_type, num_years
    )

    rows, cols = new_reports_df.shape
    if rows == 0:
        # custom_logger.warning(f"No {report_type} found for {company}")
        return None, all_reports_df
    if not os.path.exists(company_path):
        os.makedirs(company_path)
    for index, row in new_reports_df.iterrows():
        sec_url = row["filingUrl"]
        # first check if the file exists inside company_path
        filename = sec_url.split("/")[-1]
        if as_pdf:
            filename = os.path.join(company_path, filename.split(".")[0] + ".pdf")
        else:
            filename = os.path.join(
                company_path, filename + "l"
            )  # adding l to make it html and not htm
        if not os.path.exists(filename):
            # custom_logger.info(f"Downloading {filename} for {company}")
            report = download_sec_report(sec_url, as_pdf=as_pdf)
            if report and as_pdf:
                with open(filename, "wb") as f:
                    f.write(report.content)
            elif report and not as_pdf:
                with open(filename, "wb") as f:
                    f.write(report)
            else:
                pass
                # custom_logger.warning(f"Could not download {filename}")

    return new_reports_df, all_reports_df


def index_includes_report(index, report):
    # ideally you should be able to ask the index if it includes a report like below
    # but for now I am going to create a dataframe that includes the reports that
    # are already in the index
    # query_engine = new_index.as_query_engine(
    #     include_text=False,
    #     response_mode="tree_summarize"
    # )
    # response = query_engine.query(
    #     "Example Report",
    # )
    # If the report has been added to the index, the response object will contain
    # information about the report, such as its keywords and relevant triplets. If the
    # report has not been added to the index, the response object will be empty.
    pass


def prepare_data_for_chatbot(as_pdf=True, num_years=5):
    company_list = [
        "AAPL",
        "INTC",
        # "MSFT",
        # "AMZN",
        # "NVDA",
        # "GOOGL",
        # "META",
        # "BRK.B",
        # "TSLA",
        # "UNH",
    ]
    if not os.path.exists(os.path.join(current_dir, "storage", "data", "metadata.csv")):
        all_reports_df = pd.DataFrame()
    else:
        all_reports_df = pd.read_csv(
            os.path.join(current_dir, "storage", "data", "metadata.csv")
        )
    current_reports = pd.DataFrame()
    for company in company_list:
        new_reports_df, all_reports_df = prepare_sec_documents(
            company,
            all_reports_df=all_reports_df,
            as_pdf=as_pdf,
            num_years=num_years,
        )
        # save the all_reports_df to metadata.csv
        # custom_logger.info(f"Saving metadata.csv")
        all_reports_df.to_csv(
            os.path.join(current_dir, "storage", "data", "metadata.csv"), index=False
        )
        # custom_logger.info(f"Successfully saved metadata.csv")
        current_reports = pd.concat([current_reports, new_reports_df])

    newly_minted_reports = {}
    data_path = os.path.join(current_dir, "storage", "data", "sec10K")
    for index, row in current_reports.iterrows():
        company = row["ticker"]
        filename = row["filingUrl"].split("/")[-1].split(".")[0]
        if as_pdf:
            filename = os.path.join(data_path, company, filename + ".pdf")
        else:
            filename = os.path.join(data_path, company, filename + ".html")
        # append to the company if it is not already there
        newly_minted_reports[company] = newly_minted_reports.get(company, []) + [
            filename
        ]
    return newly_minted_reports


def chat_bot_agent_langchain(persist_dir="chroma_db"):
    persist_dir = os.path.join(current_dir, "storage", persist_dir)
    new_reports = prepare_data_for_chatbot(as_pdf=True, num_years=2)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    embeddings = OpenAIEmbeddings()

    companies = []
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        db = None
    for company in new_reports:
        companies.append(company)
        for file in new_reports[company]:
            loader = PyPDFLoader(file)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            # insert company metadata into each company
            if db:
                db.add_documents(texts)
            else:
                db = Chroma.from_documents(
                    texts, embeddings, persist_directory=persist_dir
                )

    db.persist()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    chat_history = []
    query = "who is ceo of apple"
    result = qa({"question": query, "chat_history": chat_history})
    print("result['answer'] = ", result["answer"])

    chat_history = [(query, result["answer"])]
    query = "Who is the ceo of Intel"
    result = qa({"question": query, "chat_history": chat_history})
    print("Question = ", query)
    print(result["answer"])

    chat_history = [(query, result["answer"])]
    query = "Intel's revenue in the past couple of years"
    result = qa({"question": query, "chat_history": chat_history})
    print("Question = ", query)
    print(result["answer"])

    chat_history = [(query, result["answer"])]
    query = "Apple's revenue in the past couple of years"
    result = qa({"question": query, "chat_history": chat_history})
    print("Question = ", query)
    print(result["answer"])


def chat_bot_agent_llama():
    new_reports = prepare_data_for_chatbot(as_pdf=False)
    storage_path = os.path.join(current_dir, "storage", "index")
    service_context = ServiceContext.from_defaults(chunk_size=512)
    UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)

    loader = UnstructuredReader()
    doc_set = {}
    all_docs = []
    companies = []
    for company in new_reports:
        companies.append(company)
        for html_file in new_reports[company]:
            document = loader.load_data(file=html_file, split_documents=False)
            # insert company metadata into each company
            for d in document:
                d.extra_info = {"company": company}
            doc_set[company] = doc_set.get(company, []) + [document]
            all_docs.extend(document)

    # set up vector indices for each company
    # custom_logger.info(f"Setting up vector indices for each company")
    service_context = ServiceContext.from_defaults(chunk_size=512)
    for company in companies:
        storage_context = StorageContext.from_defaults()
        for i, report in enumerate(doc_set[company]):
            if i == 0:
                # first check if the index already exists in storage_path/company path
                if os.path.exists(f"{storage_path}/{company}"):
                    # Overwrite the storage context if it already exists
                    storage_context = StorageContext.from_defaults(
                        persist_dir=f"{storage_path}/{company}"
                    )
                    # custom_logger.info(
                    #     "Index already exists. Loading from disk and adding new documents"
                    # )
                    cur_index = load_index_from_storage(storage_context=storage_context)
                    for doc in report:
                        cur_index.insert(doc)
                else:
                    cur_index = VectorStoreIndex.from_documents(
                        report,
                        service_context=service_context,
                        storage_context=storage_context,
                    )
            else:
                for doc in report:
                    cur_index.insert(doc)
        storage_context.persist(persist_dir=f"{storage_path}/{company}")
    # custom_logger.info(f"Finished setting up vector indices for each company")
    # Load indices from disk. This also includes the most recent indexes
    index_set = {}
    # custom_logger.info(f"Loading indices from disk")
    for company in companies:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{storage_path}/{company}"
        )
        cur_index = load_index_from_storage(storage_context=storage_context)
        index_set[company] = cur_index
    # custom_logger.info(f"Finished loading indices from disk")

    # Composing a Graph to Synthesize Answers Across 10-K Filings
    # describe each index to help traversal of composed graph
    index_summaries = [
        f"10-k Filing and financial report for {company} for years 2019 through 2022"
        for company in companies
    ]

    # define an LLMPredictor set number of output tokens
    # max_tokens=-1 will return all tokens
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, max_tokens=-1, model_name="gpt-4", client=None)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults()

    # define a list index over the vector indices
    # allows us to synthesize information across each index

    # custom_logger.info("building graph")
    graph = ComposableGraph.from_indices(
        ListIndex,
        [index_set[y] for y in companies],
        index_summaries=index_summaries,
        service_context=service_context,
        storage_context=storage_context,
    )
    # custom_logger.info("graph built")
    root_id = graph.root_id

    # [optional] save to disk
    storage_context.persist(persist_dir=f"{storage_path}/root")

    # [optional] load from disk, so you don't need to build graph from scratch
    # custom_logger.info("loading graph")
    graph = load_graph_from_storage(
        root_id=root_id,
        service_context=service_context,
        storage_context=storage_context,
    )
    # custom_logger.info("graph loaded")

    ### Setting up the Tools + Langchain Chatbot Agent
    decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)
    # define custom retrievers
    custom_query_engines = {}
    for index in index_set.values():
        query_engine = index.as_query_engine()
        query_engine = TransformQueryEngine(
            query_engine,
            query_transform=decompose_transform,
            transform_extra_info={"index_summary": index.index_struct.summary},
        )
        custom_query_engines[index.index_id] = query_engine
    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
        response_mode="tree_summarize",
        verbose=True,
    )
    # construct query engine
    graph_query_engine = graph.as_query_engine(
        custom_query_engines=custom_query_engines
    )

    # tool config
    graph_config = IndexToolConfig(
        query_engine=graph_query_engine,
        name=f"Graph Index",
        description="Useful for when you want to answer queries that require analyzing multiple financial reports for different companies. When someone asks you question about specific company, don't assume anything. Go and read the report and find the answer. If you suspect the answer should be a number then give your answers in the form of a number",
        tool_kwargs={"return_direct": True},
    )

    # define toolkit
    index_configs = []
    for company in companies:
        query_engine = index_set[company].as_query_engine(
            similarity_top_k=3,
        )

        tool_config = IndexToolConfig(
            query_engine=query_engine,
            name=f"Vector Index {company}",
            description=f"useful for when you want to answer queries about the {company} financial reports. When someone asks you question about specific company, don't assume anything. Go and read the report and find the answer. If you suspect the answer should be a number then give your answers in the form of a number",
            tool_kwargs={"return_direct": True},
        )
        index_configs.append(tool_config)

    toolkit = LlamaToolkit(
        index_configs=index_configs + [graph_config],
    )

    # custom_logger.info("Building conversation buffer memory")
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", client=None)
    agent_chain = create_llama_chat_agent(
        toolkit, llm, memory=memory, verbose=True, handle_parsing_errors=True
    )
    # custom_logger.info("Chatbot agent created")
    return agent_chain


chat_bot_agent_langchain()

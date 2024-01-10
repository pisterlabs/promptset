from typing import List, Dict, Any
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain.schema import LLMResult
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pydantic import BaseModel, Field
from langchain.agents import Tool
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import numpy as np
import sqlite3
import pandas as pd
import pinecone
import time
import os


class SearchDataBase(BaseModel):
    query: str = Field(
        description="User's request to be answered by connecting to a database and getting relevant data."
    )


class SearchDocuments(BaseModel):
    query: str = Field(
        description="User's question to be answered using existing vector database."
    )


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running"""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("*************")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*************")


def find_tools_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with {tool_name} not found")


def tool_initializer(
    names: list[str], functions: list[callable], descriptions: list[str]  # type: ignore
) -> list:
    """
    Description:
    ------------
        Allows creation of multiple Tool() instances with one method

    Arg(s):
    ------------
        name (str): Name of the tool
        function (Callable): Actual method as callable
        descriptions (str): Definition of the callable function

    Returns:
    ------------
        A list of instances of Tool object containing names, functions and descriptions

    Example:
    -----------

        ```Python

        tool_list = tool_initializer(
            names = ["DocumentAgent","SQLAgent"],
            functions=[query_from_vdb, search_from_sql],
            descriptions=[
                'Use this tool when you need to answer questions related to a funds financial risks,
                investment strategy and the invested financial instruments.
                Also you can answer general information about funds, such as the managing company.',
                'Use this tool when you need to answer questions related to a funds financial risks,
                investment strategy and the invested financial instruments.
                Also you can answer general information about funds, such as the managing company.'
            ]
        )
        ```

    """

    if len(names) != len(functions) or len(names) != len(descriptions):
        raise ValueError("Input list must have the same length")

    tools = []

    for name, func, description in zip(names, functions, descriptions):
        tool = Tool(name=name, func=func, description=description)
        tools.append(tool)

    return tools


def chunk_up_documents(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    index_name: str = "kap-documents",
):
    """
    Description:
    ------------

    Process and chunk up documents from a file, creating a vectorized database in a Pinecone index.

    Parameters:
    ------------
        file_path (str): The path to the file containing text documents to be indexed.
        chunk_size (int): The size of text chunks for processing.
        chunk_overlap (int): The amount of overlap between text chunks.
        index_name (str): The name of the Pinecone index to create or use.

    Note:
    ------------
        This function processes the file, splits it into text chunks, and indexes them in a Pinecone database.

    Example:
    ------------
    ```Python
            docs = chunk_up_documents(os.environ.get("pdf_path"))
            retriever = docs.as_retriever()
            search_result = retriever.get_relevant_documents(
                "what are the financial risks of Albaraka Portföy Kısa Vadeli Katılım Serbest (TL) fund?",
            k=2
            )
            print(search_result[0].page_content)
    ```
    """

    pinecone.init(
        api_key=os.environ.get("pinecone_api_key"),  # type: ignore
        environment=os.environ.get("pinecone_environment_value"),  # type: ignore
    )
    print("accessing to pdf directory...")
    documents = []
    for file in os.listdir(file_path):
        if file.endswith(".pdf"):
            pdf_path = file_path + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    print("splitting in progress...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators="\n\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.split_documents(documents)
    print("chunking is complete...")
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-ada-002"  # type: ignore
    )
    print("uploading embedding chunks to pinecone...")
    docsearch = Pinecone.from_documents(
        chunked_docs, embeddings_model, index_name=index_name
    )

    return docsearch


def create_vectordb(index_name: str, metric: str, dimension: int):
    """
    Description:
    ------------

    Creates a vectorized database in an existing index on Pinecone, or creates a new one if it doesn't exist.

    Arg(s):
    ------------
    index_name (str): The name of the Pinecone index to create or use.
    metric (str): The metric to be used for similarity computation (e.g., 'cosine', 'euclidean').
    dimension (int): The number of dimensions for the vectors in the index.

    Raises:
    ------------
    PineconeException: If there is an issue with the Pinecone API or Environment Value.

    Note:
    ------------
    This function initializes the Pinecone environment, creates an index if it doesn't exist, and configures the index with the specified metric and dimension.
    """
    pinecone.init(
        api_key=os.environ.get("pinecone_api_key"),  # type: ignore
        environment=os.environ.get("pinecone_environment_value"),  # type: ignore
    )

    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

    pinecone.create_index(name=index_name, metric=metric, dimension=dimension)

    return pinecone


def filter_embeddings(
    search_object, embedding_model, s_threshold: float = 0.5, r_threshold: float = 0.76
):
    """
    Description:
    ------------
        Create a pipeline for embedding filters on both embedding and redundant document similarity thresholds.

    Arg(s):
    ------------
        search_object (Pinecone.object):
        embedding_model (CohereEmbedding.object):

    Returns:
    ------------

    Usage:
    ------------
    ```Python
            # init the embedding model, 'embed-multilingual-v2.0' model can be used for languages other than English
            embeddings_cohere = CohereEmbeddings(
                cohere_api_key=os.environ.get("cohere_api"), model="embed-multilingual-v2.0"
            )
            # init the vector database instance, Pinecone
            pinecone.init(
                api_key=os.environ.get("pinecone_api_key"),
                environment=os.environ.get("pinecone_environment_value"),
            )
            # init the Pinecone searcher
            searcher = Pinecone.from_existing_index(
                index_name=os.environ.get("pinecone_index"), embedding=embeddings_cohere
            )
            # create compression retriever using filter_embeddings()
            compression_retriever = filter_embeddings(
                search_object=searcher, embedding_model=embeddings_cohere
            )
            # Initialize the RetrievalQA Chain using context compressor embedding pipeline with similarity filters
            chain = RetrievalQA.from_chain_type(
                llm=chat_model,
                chain_type="stuff",
                retriever=compression_retriever,
            )

    ```

    """
    # Embedding Filters
    relevancy_filter = EmbeddingsFilter(
        embeddings=embedding_model, s_threshold=s_threshold # type: ignore
    )

    # Add redundant filter to omit out documents with a similarity score lower than the threshold
    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=embedding_model, r_threshold=r_threshold # type: ignore
    )

    # Create a compressor pipeline
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevancy_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=search_object.as_retriever()
    )

    return compression_retriever


def scroll_down(driver):
    """
    Description:
    -------------
        Scroll down to bottom of the page.

    Args:
    -------------
        driver(method): driver method that is instantiated with webdriver object

    """
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")


def change_record_displayed(driver, table_selection: str):
    """
    Description:
    -------------
    Change number of records displayed to max available point.

    Arg(s):
    -------------
    driver(method): driver method that is instantiated with webdriver object
    table_selection(str): data table name to be directed to driver object as target. Selection criterion is based on the following conditions\n
        * table_selection == "return_based"\n
        * table_selection == "fee_based"\n
        * table_selection == "size_based"


    """
    if table_selection == "return_based":
        dropdown_selection = Select(
            driver.find_element(
                By.XPATH,
                '//div[@class="dataTables_length"]//select[@name="table_fund_returns_length"]',
            )
        )
        dropdown_selection.select_by_visible_text("250")

    elif table_selection == "fee_based":
        dropdown_selection = Select(
            driver.find_element(
                By.XPATH,
                '//div[@class="dataTables_length"]//select[@name="table_management_fees_length"]',
            )
        )
        dropdown_selection.select_by_visible_text("250")

    elif table_selection == "size_based":
        dropdown_selection = Select(
            driver.find_element(
                By.XPATH,
                '//div[@class="dataTables_length"]//select[@name="table_fund_sizes_length"]',
            )
        )
        dropdown_selection.select_by_visible_text("250")


def scrape_return_based_data(driver):
    """
    Description:
    -------------
        Scrapes return-based TEFAS funds data from 'https://www.tefas.gov.tr/FonKarsilastirma.aspx'. Scrapable data are as follows;\n
        * Fund Code
        * Fund Name\n
        * Rainbow Fund Type\n
        * 1-month % return\n
        * 3-month % return\n
        * 6-month % return\n
        * Return since year-beginning\n
        * 1-year % return\n
        * 3-year % return\n
        * 5-year % return\n

    Arg(s):
    -------------
        driver(method): driver method that is instantiated with webdriver object

    Retuns:
    -------------
        Returns a dataframe consists of abovementioned fund-related data.

    """
    variable_names = [
        "Fund_Code",
        "Fund_Name",
        "Rainbow_Fund_Type",
        "monthly_return",
        "monthly_3_return",
        "monthly_6_return",
        "since_jan",
        "annual_1_return",
        "annual_3_return",
        "annual_5_return",
    ]

    return_based_df = {var_name: [] for var_name in variable_names}

    dropdown_button = driver.find_element(By.XPATH, '//div[@id="tabs-1"]')
    time.sleep(2)
    change_record_displayed(dropdown_button, "return_based")

    wait = WebDriverWait(driver, 10)
    from_main_table_element = wait.until(
        EC.visibility_of_element_located(
            (By.XPATH, '//table[@id="table_fund_returns"]')
        )
    )
    iterable_rows = from_main_table_element.find_elements(By.XPATH, ".//tbody//tr")
    pagination_element = driver.find_elements(
        By.XPATH, '//body//div[@id="table_fund_returns_paginate"]//span//a'
    )

    last_page = int(pagination_element[-2].text)
    current_page = 1
    scroll_down(driver)

    while current_page < last_page:
        from_main_table_element = wait.until(
            EC.visibility_of_element_located(
                (By.XPATH, '//table[@id="table_fund_returns"]')
            )
        )
        iterable_rows = from_main_table_element.find_elements(By.XPATH, ".//tbody//tr")
        time.sleep(2)

        for rows in iterable_rows:
            return_based_df["Fund_Code"].append(
                rows.find_element(By.XPATH, ".//td[1]").text
            )
            return_based_df["Fund_Name"].append(
                rows.find_element(By.XPATH, ".//td[2]").text
            )
            return_based_df["Rainbow_Fund_Type"].append(
                rows.find_element(By.XPATH, ".//td[3]").text
            )
            return_based_df["monthly_return"].append(
                rows.find_element(By.XPATH, ".//td[4]").text
            )
            return_based_df["monthly_3_return"].append(
                rows.find_element(By.XPATH, ".//td[5]").text
            )
            return_based_df["monthly_6_return"].append(
                rows.find_element(By.XPATH, ".//td[6]").text
            )
            return_based_df["since_jan"].append(
                rows.find_element(By.XPATH, ".//td[7]").text
            )
            return_based_df["annual_1_return"].append(
                rows.find_element(By.XPATH, ".//td[8]").text
            )
            return_based_df["annual_3_return"].append(
                rows.find_element(By.XPATH, ".//td[9]").text
            )
            return_based_df["annual_5_return"].append(
                rows.find_element(By.XPATH, ".//td[10]").text
            )
        current_page = current_page + 1

        try:
            next_page_button = driver.find_element(
                By.XPATH, '//body//a[@id="table_fund_returns_next"]'
            )
            next_page_button.click()
            time.sleep(2)

        except:
            break

    fund_returns = pd.DataFrame(return_based_df)

    return fund_returns


def scrape_fund_management_fee_data(driver):
    """
    Description:
    -------------
        Scrapes funds' management fee data from 'https://www.tefas.gov.tr/FonKarsilastirma.aspx'. Scrapable data are as follows;\n
        * Fund Code
        * Fund Name\n
        * Rainbow Fund Type\n
        * applied_management_fee\n
        * bylaw_managemenet_fee\n
        * annual_realized_return_rate\n
        * max_total_expense_ratio\n

    Parameter(s):
    -------------
        driver(method): driver method that is instantiated with webdriver object

    Retuns:
    -------------
        Returns a dataframe consists of abovementioned fund-related data.

    """
    variable_names = [
        "applied_management_fee",
        "bylaw_management_fee",
        "annual_realized_return_rate",
        "max_total_expense_ratio",
    ]

    management_fees_df = {var_name: [] for var_name in variable_names}

    dropdown_button = driver.find_element(By.XPATH, '//div[@id="tabs-2"]')
    change_record_displayed(dropdown_button, "fee_based")
    time.sleep(2)

    from_main_table_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located(
            (By.XPATH, '//table[@id="table_management_fees"]')
        )
    )
    iterable_rows = from_main_table_element.find_elements(By.XPATH, ".//tbody//tr")
    pagination_element = driver.find_elements(
        By.XPATH, '//body//div[@id="table_management_fees_paginate"]//span//a'
    )

    last_page = int(pagination_element[-2].text)
    current_page = 1
    scroll_down(driver)

    while current_page < last_page:
        from_main_table_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, '//table[@id="table_management_fees"]')
            )
        )
        iterable_rows = from_main_table_element.find_elements(By.XPATH, ".//tbody//tr")
        time.sleep(2)

        for rows in iterable_rows:
            management_fees_df["applied_management_fee"].append(
                rows.find_element(By.XPATH, ".//td[4]").text
            )
            management_fees_df["bylaw_management_fee"].append(
                rows.find_element(By.XPATH, ".//td[5]").text
            )
            management_fees_df["annual_realized_return_rate"].append(
                rows.find_element(By.XPATH, ".//td[6]").text
            )
            management_fees_df["max_total_expense_ratio"].append(
                rows.find_element(By.XPATH, ".//td[7]").text
            )
        current_page = current_page + 1

        try:
            next_page_button = driver.find_element(
                By.XPATH, '//body//a[@id="table_management_fees_next"]'
            )
            next_page_button.click()
            time.sleep(2)

        except:
            break

    management_fee = pd.DataFrame(management_fees_df)
    return management_fee


def scrape_asset_size_data(driver):
    """
    Description:
    -------------
        Scrapes funds' asset size data from 'https://www.tefas.gov.tr/FonKarsilastirma.aspx'. Scrapable data are as follows;\n

        * init_fund_size\n
        * current_fund_size\n
        * portfolio_size_change\n
        * init_out_shares\n
        * current_out_shares\n
        * change_in_nshares\n
        * realized_return_rate\n

    Parameter(s):
    -------------
        driver(method): driver method that is instantiated with webdriver object

    Retuns:
    -------------
        Returns a dataframe consists of abovementioned fund-related data.

    """
    variable_names = [
        "init_fund_size",
        "current_fund_size",
        "portfolio_size_change",
        "init_out_shares",
        "current_out_shares",
        "change_in_nshares",
        "realized_return_rate",
    ]

    fund_asset_sizes_df = {var_name: [] for var_name in variable_names}

    dropdown_button = driver.find_element(By.XPATH, '//div[@id="tabs-3"]')
    change_record_displayed(dropdown_button, "size_based")

    from_main_table_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, '//table[@id="table_fund_sizes"]'))
    )
    iterable_rows = from_main_table_element.find_elements(By.XPATH, ".//tbody//tr")
    pagination_element = driver.find_elements(
        By.XPATH, '//body//div[@id="table_fund_sizes_paginate"]//span//a'
    )

    last_page = int(pagination_element[-2].text)
    current_page = 1
    scroll_down(driver)

    time.sleep(2)
    while current_page < last_page:
        from_main_table_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, '//table[@id="table_fund_sizes"]')
            )
        )
        iterable_rows = from_main_table_element.find_elements(By.XPATH, ".//tbody//tr")
        time.sleep(2)

        for rows in iterable_rows:
            fund_asset_sizes_df["init_fund_size"].append(
                rows.find_element(By.XPATH, ".//td[4]").text
            )
            fund_asset_sizes_df["current_fund_size"].append(
                rows.find_element(By.XPATH, ".//td[5]").text
            )
            fund_asset_sizes_df["portfolio_size_change"].append(
                rows.find_element(By.XPATH, ".//td[6]").text
            )
            fund_asset_sizes_df["init_out_shares"].append(
                rows.find_element(By.XPATH, ".//td[7]").text
            )
            fund_asset_sizes_df["current_out_shares"].append(
                rows.find_element(By.XPATH, ".//td[8]").text
            )
            fund_asset_sizes_df["change_in_nshares"].append(
                rows.find_element(By.XPATH, ".//td[9]").text
            )
            fund_asset_sizes_df["realized_return_rate"].append(
                rows.find_element(By.XPATH, ".//td[10]").text
            )

        current_page = current_page + 1

        try:
            next_page_button = driver.find_element(
                By.XPATH, '//body//a[@id="table_fund_sizes_next"]'
            )
            next_page_button.click()
            time.sleep(2)

        except:
            break

    asset_size_df = pd.DataFrame(fund_asset_sizes_df)

    return asset_size_df


def scrape_fund_details():
    url = "https://fundturkey.com.tr/(S(whrurtm4qvulf2vgismdtgax))/TarihselVeriler.aspx"
    service = Service(executable_path=os.environ.get("web_driver_path")) # type: ignore
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    driver.maximize_window()

    # Date selector
    main_date_table = driver.find_element(By.XPATH, '//div[@class="dates historic"]')

    # if datetime.today() == weekend; todays_date = datetime.today() - timedelta(days=2)
    if datetime.today().weekday() in [5, 6]:
        todays_date = datetime.today() - timedelta(days=2)
        parsed_date = todays_date.strftime("%d.%m.%Y")
        start_date = WebDriverWait(main_date_table, 2).until(
            EC.presence_of_element_located(
                (By.XPATH, '//input[@name="ctl00$MainContent$TextBoxStartDate"]')
            )
        )
        end_date = WebDriverWait(main_date_table, 2).until(
            EC.presence_of_element_located(
                (By.XPATH, '//input[@name="ctl00$MainContent$TextBoxEndDate"]')
            )
        )

        if start_date and end_date is not None:
            driver.execute_script(
                "arguments[0].value = arguments[1]", start_date, parsed_date
            )
            time.sleep(1)
            driver.execute_script(
                "arguments[0].value = arguments[1]", end_date, parsed_date
            )
            time.sleep(1)

    # if datetime.today() != weekend; todays_date = datetime.today()
    else:
        todays_date = datetime.today()
        parsed_date = todays_date.strftime("%d.%m.%Y")
        start_date = WebDriverWait(main_date_table, 2).until(
            EC.presence_of_element_located(
                (By.XPATH, '//input[@name="ctl00$MainContent$TextBoxStartDate"]')
            )
        )
        end_date = WebDriverWait(main_date_table, 2).until(
            EC.presence_of_element_located(
                (By.XPATH, '//input[@name="ctl00$MainContent$TextBoxEndDate"]')
            )
        )

        if start_date and end_date is not None:
            driver.execute_script(
                "arguments[0].value = arguments[1]", start_date, parsed_date
            )
            time.sleep(1)
            driver.execute_script(
                "arguments[0].value = arguments[1]", end_date, parsed_date
            )
            time.sleep(1)

    # Portfolio Breakdown tab selector
    clickable_pb_tab = WebDriverWait(driver, 2).until(
        EC.presence_of_element_located(
            (By.XPATH, '//ul[@role="tablist"]//li[@aria-labelledby="ui-id-2"]')
        )
    )

    if clickable_pb_tab is not None:
        clickable_pb_tab.click()
        time.sleep(5)

    # Click on View button
    clicklable_view_button = WebDriverWait(main_date_table, 2).until(
        EC.presence_of_element_located((By.XPATH, "//input[@value='View']"))
    )

    if clicklable_view_button is not None:
        clicklable_view_button.click()
        time.sleep(3)

    # source of the main table where available rows will be scraped
    source_element = WebDriverWait(driver, 2).until(
        EC.presence_of_element_located(
            (By.XPATH, '//div[@class="dataTables_scrollBody"]')
        )
    )

    # click on 'record-show' dropdown where; text_value=='250'
    view_dropdown = Select(
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//select[@name="table_allocation_length"]',
                )
            )
        )
    )
    view_dropdown.select_by_visible_text("250")
    time.sleep(1)
    scroll_down(driver)
    time.sleep(1)

    if source_element:
        iterable_rows = WebDriverWait(source_element, 2).until(
            EC.presence_of_all_elements_located((By.XPATH, ".//tbody//tr"))
        )
        pagination_element = WebDriverWait(driver, 2).until(
            EC.presence_of_all_elements_located(
                (
                    By.XPATH,
                    '//div[@class="dataTables_paginate paging_simple_numbers"]//span//a',
                )
            )
        )

        columns = [
            "Date",
            "Fund Code",
            "Fund Title",
            "Stock (%)",
            "Government Bond (%)",
            "Treasury Bill (%)",
            "Government Currency Debt Sec. (%)",
            "Commercial Paper (%)",
            "Private Sector Bond (%)",
            "Asset-Backed Securities (%)",
            "Government Bonds and Bills (FX) (%)",
            "International Corporate Debt Sec. (%)",
            "Takasbank Money Market (%)",
            "Government Lease Certificate (TL) (%)",
            "Government Lease Certificate (FC) (%)",
            "Private Sector Lease Certificates (%)",
            "International Gov. Lease Certificates (%)",
            "International Cor. Lease Certificates (%)",
            "Deposit Account (Turkish Lira) (%)",
            "Deposit Account (FC) (%)",
            "Deposit Account (Gold) (%)",
            "Participation Account (TL) (%)",
            "Participation Account (FC) (%)",
            "Participation Account (Gold) (%)",
            "Repo (%)",
            "Reverse-Repo (%)",
            "Precious Metals (%)",
            "ETF Issued in Precious Metals (%)",
            "Government Debt Sec. Issued in Precious Metal (%)",
            "Gov. Lease Certificates Issued in Precious Metal (%)",
            "International Government Debt Sec. (%)",
            "International Corporate Debt Sec. (%)",
            "Foreign Equity (%)",
            "International ETF (%)",
            "Investment Funds Participation Share (%)",
            "Exchange Traded Fund Participation Share (%)",
            "Real Estate I. Fund Participation Share (%)",
            "Venture Capital I. Fund Participation Share (%)",
            "Futures Contract Cash Collateral (%)",
            "Other (%)",
        ]

        # Initialize dictionary to store data
        data = {column: [] for column in columns}

        current_page = 1
        last_page = int(pagination_element[-1].text)

        while current_page <= last_page:
            datatable_source = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//table[@id="table_allocation"]//tbody')
                )
            )

            iterable_data = WebDriverWait(datatable_source, 2).until(
                EC.presence_of_all_elements_located((By.XPATH, ".//tr[@class]"))
            )

            for row in iterable_data:
                for i, column in enumerate(columns):
                    data[column].append(
                        row.find_element(By.XPATH, f".//td[{i + 1}]").text
                    )

            current_page += 1
            try:
                next_page_button = driver.find_element(
                    By.XPATH,
                    '//div[@id="table_allocation_wrapper"]//div[@class="dataTables_paginate paging_simple_numbers"]//a[@class="paginate_button next"]',
                )
                next_page_button.click()
                time.sleep(2)
            except:
                break

        combined_data = [dict(zip(columns, row)) for row in zip(*data.values())]

        driver.quit()
    unprocessed_df = pd.DataFrame(combined_data)
    processed_df = process_columns(unprocessed_df, unprocessed_df.iloc[:, 3:].columns)
    combined_df = pd.concat([unprocessed_df.iloc[:, :3], processed_df], axis=1)

    return combined_df


def process_columns(df, column_list):
    new_df = pd.DataFrame()

    for col_name in column_list:
        new_list = []
        for val in df[col_name]:
            if isinstance(val, str):
                if (val.count(",") == 1) and (val.count(".") == 1):
                    val = val.replace(".", ",", 1)
                    val = val.replace(",", "", (val.count(",") - 1))
                    val = val.replace(",", ".")
                    new_list.append(float(val))

                elif (val.count(".") > 1) and (val.count(",") == 1):
                    val = val.replace(".", "", (val.count(".")))
                    val = val.replace(",", ".")
                    new_list.append(float(val))

                elif val.count(",") == 1:
                    val = val.replace(",", ".")
                    new_list.append(float(val))

                elif val.count(",") > 1:
                    val = val.replace(",", "", (val.count(",") - 1))
                    val = val.replace(",", ".")
                    new_list.append(float(val))

                elif val == "-":
                    new_list.append(np.nan)

                else:
                    new_list.append(val)
            else:
                new_list.append(val)

        new_df[col_name] = new_list
    print(new_df)
    return new_df


def create_table(connection, cursor):
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tefastable(
                fund_code TEXT,
                fund_name TEXT,
                fund_type TEXT,
                monthly_return FLOAT,
                monthly_3_return FLOAT,
                monthly_6_return FLOAT,
                since_jan FLOAT,
                annual_1_return FLOAT,
                annual_3_return FLOAT,
                annual_5_return FLOAT,
                applied_management_fee FLOAT,
                bylaw_management_fee FLOAT,
                annual_realized_return_rate FLOAT,
                max_total_expense_ratio FLOAT,
                initial_fund_size INT,
                current_fund_size INT,
                portfolio_size_change FLOAT,
                initial_number_of_shares INT,
                current_number_of_shares INT,
                change_in_shares FLOAT,
                realized_return_rate FLOAT
                )
            
            """
        )
        connection.commit()
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")


def create_tablev2(connection, cursor, tablename: str):
    try:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {tablename}(
            date TEXT,
            fund_code TEXT,
            fund_title TEXT,
            stock_percentage FLOAT,
            government_bond_percentage FLOAT,
            treasury_bill_percentage FLOAT,
            government_currency_debt_percentage FLOAT,
            commercial_paper_percentage FLOAT,
            private_sector_bond_percentage FLOAT,
            asset_backed_securities_percentage FLOAT,
            government_bonds_and_bills_fx_percentage = FLOAT,
            international_corporate_debt_percentage FLOAT,
            takasbank_money_market_percentage FLOAT,
            government_lease_certificate_tl_percentage FLOAT,
            government_lease_certificate_fc_percentage FLOAT,
            private_sector_lease_certificates_percentage FLOAT,
            international_gov_lease_certificates_percentage FLOAT,
            international_cor_lease_certificates_percentage FLOAT,
            deposit_account_tl_percentage FLOAT,
            deposit_account_fc_percentage FLOAT,
            deposit_account_gold_percentage FLOAT,
            participation_account_tl_percentage FLOAT,
            participation_account_fc_percentage FLOAT,
            participation_account_gold_percentage FLOAT,
            repo_percentage FLOAT,
            reverse_repo_percentage FLOAT,
            precious_metals_percentage FLOAT,
            etf_issued_in_precious_metals_percentage FLOAT,
            government_debt_sec_issued_in_precious_metal_percentage FLOAT,
            gov_lease_certificates_issued_in_precious_metal_percentage FLOAT,
            international_government_debt_percentage FLOAT,
            foreign_equity_percentage FLOAT,
            international_etf_percentage FLOAT,
            investment_funds_participation_share_percentage FLOAT,
            etf_participation_share_percentage FLOAT,
            real_estate_fund_participation_share_percentage FLOAT,
            venture_capital_fund_participation_share_percentage FLOAT,
            futures_contract_cash_collateral_percentage FLOAT,
                )
            
            """
        )
        connection.commit()
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")


def add_data(table_name: str, connection, data: pd.DataFrame):
    try:
        data.to_sql(table_name, connection, if_exists="replace", index=False)

    except Exception as e:
        print(f"Error inserting data: {e}")


def get_data(cursor, table_name: str):
    cursor.execute(f"SELECT * From {table_name};")
    datatable = cursor.fetchall()

    return datatable


from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.sql_database import SQLDatabase

db = SQLDatabase.from_uri(database_uri=os.environ.get("uri_path"))
memory = ConversationBufferMemory(return_messages=True)


def get_schema(_):
    global db
    return db.get_table_info()


def run_query(query):
    global db
    return db.run(query)


def save(input_output):
    global memory
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]

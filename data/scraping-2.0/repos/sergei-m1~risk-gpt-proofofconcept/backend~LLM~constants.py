import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

INGEST_THREADS = 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
}

# Instructor Large: "hkunlp/instructor-large"
# Instructor XL: "hkunlp/instructor-xl"
EMBEDDINGS_NAME = "hkunlp/instructor-xl"

OPENAI_LLM_NAME = "gpt-3.5-turbo"

N_DOCUMENTS_SEARCH = 4


DOCUMENTS = {
    "draft_prudential_standard_aps_117_capital_adequacy_interest_rate_risk_in_the_banking_book.pdf": "https://www.apra.gov.au/sites/default/files/draft_prudential_standard_aps_117_capital_adequacy_interest_rate_risk_in_the_banking_book.pdf",
    "Interestrateriskinthebankingbook_basel2016.pdf": "https://www.bis.org/bcbs/publ/d368.pdf",
    "irrbb-revisited-nov-2019.pdf": "https://assets.kpmg.com/content/dam/kpmg/au/pdf/2019/irrbb-revisited-nov-2019.pdf",
    "KevinDavidIRRBB.pdf": "https://www.kevindavis.com.au/BankingBook/22-%20IRRBB.pdf",
    "lu-interest-rate-risk-banking-book-18012018.pdf": "https://www2.deloitte.com/content/dam/Deloitte/dk/Documents/risk/lu-interest-rate-risk-banking-book-18012018.pdf",
    "APG-117-January 2008_0.pdf": "https://www.apra.gov.au/sites/default/files/APG-117-January%202008_0.pdf",
    "IRRBBriskmanagementstrategies.pdf": "https://www.efrag.org/Assets/Download?assetUrl=%2Fsites%2Fwebpublishing%2FMeeting%20Documents%2F2006231239205943%2F06-03%20DRM%20IRRBB%20EFRAG%20TEG%2021-05-19.pdf",
    "EBAIRRBBguidelines.pdf": "https://www.eba.europa.eu/sites/default/documents/files/documents/10180/2282655/169993e1-ad7a-4d78-8a27-1975b4860da0/Guidelines%20on%20the%20management%20of%20interest%20rate%20risk%20arising%20from%20non-trading%20activities%20%28EBA-GL-2018-02%29.pdf?retry=1",
    "StochasticApproachIRRBB.pdf": "https://www.researchgate.net/publication/320939054_A_Stochastic_Approach_to_the_Measurement_of_Interest_Rate_Risk_in_the_Banking_Book",
}

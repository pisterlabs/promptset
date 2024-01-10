from datetime import datetime
from pathlib import Path
import pandas as pd
import unicodedata

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from tqdm import tqdm

from whoosh import scoring
from whoosh.fields import DATETIME, ID, STORED, TEXT, Schema, KEYWORD
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser, syntax
from whoosh.qparser.plugins import MultifieldPlugin


from artefacts import (
    ALL_CASES_CSV,
    NORMED_ALL_CASES_CSV,
    ASCII_ALL_CASES_CSV,
    VECTOR_INDEX_FOLDER_PATH,
    WHOOSH_INDEX_FOLDER_PATH,
)
from legalsearch.data_preprocessing import load_all_cases
from legalsearch.models import AggregatedCaseFields

MINILM_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
LEGAL_BERT_REPO_ID = "nlpaueb/legal-bert-base-uncased"


BM25_SCHEMA = Schema(
    title=TEXT(stored=True),
    summary=TEXT(stored=True),
    permalink=STORED(),
    jurisdictions=TEXT(stored=True),
    principal_laws=TEXT(stored=True),
    case_categories=TEXT(stored=True),
    status=TEXT(stored=True),
    filing_year=DATETIME(stored=True),
)


def normalize_file(raw_file, clean_file):
    with open(raw_file, "r+", encoding="utf-8") as file_reader:
        content = file_reader.read()
        ascii_content = unicodedata.normalize("NFKD", content).encode("ascii", "ignore")

    with open(clean_file, "wb") as file_writer:
        file_writer.write(ascii_content)


def prepare_for_indexing(df_path):
    df = pd.read_csv(df_path)

    df[AggregatedCaseFields.FILING_YEAR] = pd.to_datetime(
        df[AggregatedCaseFields.FILING_YEAR], format="ISO8601"
    )
    df[AggregatedCaseFields.SUMMARY] = df[AggregatedCaseFields.SUMMARY].fillna(
        "Not provided"
    )

    df[AggregatedCaseFields.STATUS] = (
        df[AggregatedCaseFields.STATUS].fillna("").astype(str)
    )
    df[AggregatedCaseFields.STATUS] = df[AggregatedCaseFields.STATUS].astype("category")
    df[AggregatedCaseFields.FILING_YEAR] = df[AggregatedCaseFields.FILING_YEAR].fillna(
        ""
    )
    df[AggregatedCaseFields.PERMALINK] = df[AggregatedCaseFields.PERMALINK].fillna("")

    df = df.astype(str).replace("nan", "")

    return df[AggregatedCaseFields.COLUMNS]


def prepare_csv_file():
    if not ALL_CASES_CSV.exists():
        load_all_cases()

    if not ASCII_ALL_CASES_CSV.exists():
        normalize_file(ALL_CASES_CSV, ASCII_ALL_CASES_CSV)

    if not NORMED_ALL_CASES_CSV.exists():
        prepare_for_indexing(ASCII_ALL_CASES_CSV).to_csv(
            NORMED_ALL_CASES_CSV, index=False
        )


def build_vector_index():
    embeddings = HuggingFaceEmbeddings(model_name=LEGAL_BERT_REPO_ID)

    if not NORMED_ALL_CASES_CSV.exists():
        prepare_csv_file()

    loader = CSVLoader(
        str(NORMED_ALL_CASES_CSV), source_column=AggregatedCaseFields.PERMALINK
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    docsearch = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name="climatecase",
        persist_directory=str(VECTOR_INDEX_FOLDER_PATH),
    )
    return docsearch


def load_vector_index():
    embeddings = HuggingFaceEmbeddings(model_name=LEGAL_BERT_REPO_ID)

    docsearch = Chroma(
        collection_name="climatecase",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_INDEX_FOLDER_PATH),
    )
    return docsearch


def build_bm25_index():
    if not WHOOSH_INDEX_FOLDER_PATH.exists():
        WHOOSH_INDEX_FOLDER_PATH.mkdir()

    if not NORMED_ALL_CASES_CSV.exists():
        prepare_csv_file()

    df = prepare_for_indexing(ASCII_ALL_CASES_CSV)

    index = create_in(WHOOSH_INDEX_FOLDER_PATH, BM25_SCHEMA)

    writer = index.writer()

    for i, row in tqdm(df.iterrows()):
        writer.add_document(
            title=row[AggregatedCaseFields.TITLE],
            summary=row[AggregatedCaseFields.SUMMARY] or "",
            permalink=row[AggregatedCaseFields.PERMALINK] or "",
            jurisdictions=", ".join(row[AggregatedCaseFields.JURISDICTIONS])
            if row[AggregatedCaseFields.JURISDICTIONS]
            else "",
            principal_laws=", ".join(row[AggregatedCaseFields.PRINCIPAL_LAWS])
            if row[AggregatedCaseFields.PRINCIPAL_LAWS]
            else "",
            case_categories=", ".join(row[AggregatedCaseFields.CATEGORIES])
            if row[AggregatedCaseFields.CATEGORIES]
            else "",
            status=row[AggregatedCaseFields.STATUS] or "",
            filing_year=row[AggregatedCaseFields.FILING_YEAR]
            if row[AggregatedCaseFields.FILING_YEAR] != "NaT"
            else datetime(1970, 1, 1),
        )
    writer.commit()


if not WHOOSH_INDEX_FOLDER_PATH.exists() or not any(WHOOSH_INDEX_FOLDER_PATH.iterdir()):
    build_bm25_index()

bm25_index = open_dir(WHOOSH_INDEX_FOLDER_PATH)
vector_index = load_vector_index()


def my_query_parser(fieldnames, schema, fieldboosts=None):
    p = QueryParser(None, schema, group=syntax.OrGroup)
    mfp = MultifieldPlugin(fieldnames, fieldboosts=fieldboosts)
    p.add_plugin(mfp)
    return p


def bm25_search(query, top_k=10):
    query_parser = my_query_parser(
        fieldnames=["title", "summary"],
        schema=BM25_SCHEMA,
    ).parse(query)

    with bm25_index.searcher(weighting=scoring.BM25F()) as searcher:
        results = searcher.search(
            q=query_parser,
            limit=top_k,
            terms=True,
        )

    doc_ids = [id for score, id in results.top_n]
    return doc_ids

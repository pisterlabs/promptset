"""Workflow to process pdfs into a vector database.

As provided here, this workflow assumes that you've looked up some set of papers
using Semantic Scholar's API, and stored them in a `papers` table with this
schema:

``` sql
CREATE TABLE papers (
        semantic_scholar_id TEXT PRIMARY KEY NOT NULL,
        semantic_scholar_json TEXT NOT NULL
    )
```

And a table `pdfs` with this schema, that contains the pdf content for each
paper:

``` sql
CREATE TABLE pdfs (
        doi TEXT PRIMARY KEY NOT NULL,
        pdf_content BLOB NOT NULL,
        pdf_md5 TEXT NOT NULL,
        direct INTEGER NOT NULL -- Treated as boolean; could we download directly from the open internet, aot UC?
    )
```

If those tables exist, the downstream sqlite tables and chromadb vector store
will be created automatically.

The sqlite database provided here at
`data/funk-etal-2008.selected-open-access.db` contains examples.
"""

import atexit
import io
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import time

import dotenv
import openai
import pandas as pd
import requests
from langchain.vectorstores import Chroma
from metaflow import FlowSpec, Parameter, step

import common as cm

dotenv.load_dotenv()

OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "No OpenAI key!"))


def create_tables(connection: sqlite3.Connection):
    """
    Creates a SQLite database with a table for storing PDFs.

    :param path: A pathlib.Path object pointing to the SQLite database file.
    """
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS parsed_papers (
            doi TEXT PRIMARY KEY NOT NULL,
            grobid_xml TEXT NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doi TEXT PRIMARY KEY NOT NULL,
            document_json TEXT NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS heading_spellcheck (
            original TEXT PRIMARY KEY NOT NULL,
            spellchecked TEXT NOT NULL
        )
    """
    )

    connection.commit()
    connection.close()

    return True


def add_xml(db: sqlite3.Connection, doi: str, xml_content: str):
    """
    Updates an existing paper record in the database with the PDF content.

    :param db: SQLite database connection object.
    :param doi: String representing the Digital Object Identifier of the paper.
    :param pdf_path: A pathlib.Path object pointing to the PDF file.
    """
    cursor = db.cursor()

    # Update the record with the PDF content
    cursor.execute(
        "INSERT INTO parsed_papers (doi, grobid_xml) VALUES (?, ?)",
        (doi, xml_content),
    )

    db.commit()


def add_document(db: sqlite3.Connection, doi: str, document: list[dict[str, str]]):
    """
    Updates an existing paper record in the database with the PDF content.

    :param db: SQLite database connection object.
    :param doi: String representing the Digital Object Identifier of the paper.
    :param pdf_path: A pathlib.Path object pointing to the PDF file.
    """
    cursor = db.cursor()

    # Update the record with the PDF content
    cursor.execute(
        "INSERT INTO documents (doi, document_json) VALUES (?, ?)",
        (doi, json.dumps(document)),
    )

    db.commit()


def spellcheck_gpt3_5(text, client=OPENAI_CLIENT, model="gpt-3.5-turbo"):
    system_prompt = """\
You are a spellchecking assistant. You will correct text extracted from
the headings of scientific paper PDFs, which may have extraction-related errors,
like added spaces or extraneous formatting. The prompt will include only the
original text, and you will reply only with the corrected text.
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=20,
        temperature=0.1,
    )

    answer = completion.choices[0].message.content.strip()

    return answer


def add_spelling_correction(db: sqlite3.Connection, original: str, corrected: str):
    """
    Adds or updates a spelling correction record in the database.

    :param db: SQLite database connection object.
    :param original: The original text.
    :param corrected: The corrected (spellchecked) text.
    """
    cursor = db.cursor()

    # Using SQLite's UPSERT feature
    cursor.execute(
        """
        INSERT INTO heading_spellcheck (original, spellchecked)
        VALUES (?, ?)
        ON CONFLICT(original)
        DO UPDATE SET spellchecked = excluded.spellchecked
        """,
        (original, corrected),
    )

    db.commit()


def spellcheck_heading(db: sqlite3.Connection, heading: str):
    """
    Checks if the heading is already spellchecked in the database.
    If not, spellchecks using spellcheck_gpt3_5(), adds to database, and returns the correction.

    :param db: SQLite database connection object.
    :param heading: The heading text to be spellchecked.
    :return: The spellchecked heading.
    """
    cursor = db.cursor()

    # Check if the heading is already in the database
    cursor.execute(
        """
        SELECT spellchecked
        FROM heading_spellcheck
        WHERE original = ?
        """,
        (heading,),
    )
    result = cursor.fetchone()

    if result:
        # Return the correction if query found in the database
        return result[0]
    else:
        # Run through spellcheck_gpt3_5() - assuming this is a defined function
        corrected_heading = spellcheck_gpt3_5(heading)

        # Add the new spellchecked heading to the database
        # logger.debug(
        #     f"Caching spellchecked heading [{heading}] to [{corrected_heading}]."
        # )
        add_spelling_correction(db, heading, corrected_heading)

        return corrected_heading


def document_generator(db: sqlite3.Connection):
    cursor = db.cursor()

    cursor.execute("SELECT * FROM documents")

    yield from cursor


def filter_metadata(md):
    return {k: v for k, v in md.items() if v is not None}


class ProcessPapersFlow(FlowSpec):
    """Create papers database."""

    db_path = Parameter("db-path", default="data/funk-etal-2008.db", type=pathlib.Path)
    skip_grobid = Parameter("skip-grobid", default=False, type=bool, is_flag=True)
    grobid_server = Parameter(
        "grobid-server", default="http://localhost:8070/api/processFulltextDocument"
    )
    vector_db_path = Parameter(
        "vector-db-path", default="data/funk-etal-2008.chromadb", type=pathlib.Path
    )
    papers_collection = Parameter("papers-collection", default="funk-etal-2008-meta")
    embeddings_model = Parameter(
        "embeddings-model", default="jinaai/jina-embedding-l-en-v1"
    )
    overwrite_vector_db = Parameter(
        "overwrite-vector-db", default=False, type=bool, is_flag=True
    )

    @step
    def start(self):
        """"""

        # Create required tables (if they don't already exist).
        db = sqlite3.connect(self.db_path)
        create_tables(db)

        self.next(self.run_grobid)

    @step
    def run_grobid(self):
        """Run GROBID academic PDF->XML parser."""
        db = sqlite3.connect(self.db_path)

        self.pdf_dois = pd.read_sql(
            """
        select
            doi
        from pdfs
        left join parsed_papers using (doi)
        where
            parsed_papers.grobid_xml is null
        """,
            db,
        ).doi.tolist()
        print(f"Found [{len(self.pdf_dois)}] pdfs to parse.")

        if self.skip_grobid or not self.pdf_dois:
            print("Skipping Grobid parsing.")
        else:
            docker_process = subprocess.Popen(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "-p",
                    "8070:8070",
                    "-p",
                    "8071:8071",
                    "grobid/grobid:0.7.3",
                ]
            )
            atexit.register(docker_process.terminate)
            time.sleep(10)

            try:
                requests.get(self.grobid_server)
            except requests.exceptions.RequestException as err:
                print("Grobid server not accepting connections, stopping.")
                print(f"{err=}")
                sys.exit()

            for doi in self.pdf_dois:
                print(f"Parsing pdf from DOI: [{doi}]")
                pdf_bytes = pd.read_sql(
                    f"select pdf_content from pdfs where doi = '{doi}'",
                    db,
                ).pdf_content.squeeze()
                print(f"Loaded [{len(pdf_bytes)}] bytes.")
                try:
                    data = {
                        "generateIDs": "1",
                        "consolidateHeader": "1",
                        "segmentSentences": "1",
                        "teiCoordinates": ["head", "s"],
                    }
                    files = {"input": io.BytesIO(pdf_bytes)}
                    r = requests.request(
                        "POST",
                        self.grobid_server,
                        headers=None,
                        params=None,
                        files=files,
                        data=data,
                        timeout=60,
                    )
                    xml_data = r.text
                except requests.exceptions.ReadTimeout:
                    print("GROBID server timed out. Return None.")
                    xml_data = None

                if xml_data is None:
                    continue

                add_xml(db, doi, xml_data)

            print("Stopping grobid server.")
            docker_process.terminate()
            docker_process.wait()

        self.next(self.process_xml)

    @step
    def process_xml(self):
        db = sqlite3.connect(self.db_path)

        self.xml_dois = pd.read_sql(
            """
        select
            doi
        from parsed_papers
        left join documents using (doi)
        where
            documents.document_json is null
        """,
            db,
        ).doi.tolist()
        print(f"Found [{len(self.xml_dois)}] xml results to process.")

        for doi in self.xml_dois:
            print(f"Processing xml from DOI: [{doi}]")
            grobid_xml = pd.read_sql(
                f"select grobid_xml from parsed_papers where doi = '{doi}'",
                db,
            ).grobid_xml.squeeze()
            print(f"Loaded XML text of length [{len(grobid_xml)}].")

            try:
                processed = cm.process_xml(grobid_xml)
            except Exception as e:
                print(f"Failed processing: [{e}]")
                continue

            add_document(db, doi, processed)

        self.next(self.build_vector_db)

    @step
    def build_vector_db(self):
        """Build vector db."""

        db = sqlite3.connect(self.db_path)

        self.doc_dois = set(pd.read_sql("select doi from documents", db).doi.tolist())

        embeddings = cm.load_embeddings_model(self.embeddings_model)

        if self.overwrite_vector_db and pathlib.Path(self.vector_db_path).exists():
            print("Deleting existing vector database.")
            shutil.rmtree(self.vector_db_path)

        vector_db = Chroma(
            persist_directory=str(self.vector_db_path),
            collection_name=self.papers_collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

        existing_docs = set(
            e["doi"] for e in vector_db.get(include=["metadatas"])["metadatas"]
        )
        print(f"Found [{len(existing_docs)}] existing docs indexed in vector db.")

        self.doc_dois = self.doc_dois - existing_docs
        print(f"Found [{len(self.doc_dois)}] unprocessed DOIs.")

        for doi in self.doc_dois:
            print(f"Loading documents for DOI: [{doi}]")
            document_json = pd.read_sql(
                f"select document_json from documents where doi = '{doi}'",
                db,
            ).document_json.squeeze()

            docs = json.loads(document_json)

            texts = [
                spellcheck_heading(db, doc["page_content"])
                if (
                    doc["metadata"]["is_heading"]
                    and (doc["metadata"]["num_tokens"] < 20)
                )
                else doc["page_content"]
                for doc in docs
            ]
            metadatas = [
                filter_metadata(doc["metadata"]) | {"doi": doi} for doc in docs
            ]
            vector_db.add_texts(texts, metadatas)

        self.next(self.end)

    @step
    def end(self):
        """"""
        pass


if __name__ == "__main__":
    ProcessPapersFlow()

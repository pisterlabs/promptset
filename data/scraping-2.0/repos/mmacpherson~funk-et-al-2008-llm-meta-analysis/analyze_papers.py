"""Workflow to conduct LLM-based meta-analysis over scientific paper PDFs.

Assumes PDFs have already been process into a vector store; see
`process_papers.py`.

For this to run, you need to enable access to OpenAI's API; create a file `.env`
in the present directory, with the following content:

```
OPENAI_API_KEY={key_here}
```

"""

import json
import os
import sqlite3
import sys
import time

import dotenv
import yaml
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from metaflow import FlowSpec, JSONType, Parameter, current, step

import common as cm
import decision_tree_chat as dtc

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


def create_table(connection: sqlite3.Connection):
    """ """
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis (
            run_id TEXT NOT NULL,
            doi TEXT NOT NULL,
            states_json TEXT NOT NULL,
            chat_json TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            UNIQUE(run_id, doi)
        )
    """
    )

    connection.commit()
    connection.close()


def to_json(e):
    try:
        return json.dumps(e)
    except TypeError:
        try:
            # Handles LangChain items.
            return json.dumps([x.to_json() for x in e])
        except AttributeError:
            # Handles pydantic items.
            return json.dumps([x.model_dump() for x in e])


def store_results(db, run_id, doi, states, messages, answers):
    """
    Stores the results of an analysis into the database.

    Args:
      db: SQLite database connection object.
      run_id: The run identifier.
      ssid: The Semantic Scholar ID.
      states: The states to be stored.
      messages: The chat messages to be stored.
      answers: The answers to be stored.
    """
    cursor = db.cursor()

    states_json = to_json(states)
    messages_json = to_json(messages)
    answers_json = to_json(answers)

    cursor.execute(
        "INSERT INTO analysis (run_id, doi, states_json, chat_json, answers_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, doi, states_json, messages_json, answers_json),
    )

    db.commit()


class AnalyzePapersFlow(FlowSpec):
    """Create papers database."""

    db_path = Parameter("db-path", default="data/funk-etal-2008.db")
    vector_db_path = Parameter("vector-db-path", default="data/funk-etal-2008.chromadb")
    tree_path = Parameter("tree-path", default="data/decision-tree.yaml")
    embeddings_model = Parameter(
        "embeddings-model", default="jinaai/jina-embedding-l-en-v1"
    )
    dois = Parameter("dois", default=None, type=JSONType)
    papers_collection = Parameter("papers-collection", default="funk-etal-2008-meta")
    model = Parameter("model", default="gpt-3.5-turbo")
    max_paper_tokens = Parameter("num-tokens", default=1200, type=int)
    temperature = Parameter("temperature", default=0.1, type=float)
    max_reprompts = Parameter("max-reprompts", default=0, type=int)

    @step
    def start(self):
        """"""

        db = sqlite3.connect(self.db_path)
        create_table(db)
        print("Created analysis table.")

        embeddings = cm.load_embeddings_model(self.embeddings_model)
        vector_db = Chroma(
            persist_directory=self.vector_db_path,
            collection_name=self.papers_collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        available_dois = set(
            e["doi"] for e in vector_db.get(include=["metadatas"])["metadatas"]
        )

        if self.dois is not None:
            missing_dois = set(self.dois) - available_dois
            if missing_dois:
                print(
                    f"Of [{len(self.dois)}] requested papers, [{len(missing_dois)}] are missing."
                )
                print(missing_dois)
            else:
                print(f"All [{len(self.dois)}] requested papers are available.")
            self.selected_dois = set(self.dois) & available_dois
        else:
            self.selected_dois = available_dois

        print(f"Working on [{len(self.selected_dois)}] papers.")

        self.next(self.analyze_papers)

    @step
    def analyze_papers(self):
        """"""

        if OPENAI_API_KEY is None:
            print("Stopping, didn't find OpenAI API key in .env.")
            sys.exit(1)

        db = sqlite3.connect(self.db_path)

        # Load decision tree.
        with open(self.tree_path) as f:
            self.decision_tree = dtc.ChatTreeConfiguration.model_validate(
                yaml.safe_load(f)
            )

        embeddings = cm.load_embeddings_model(self.embeddings_model)
        vector_db = Chroma(
            persist_directory=self.vector_db_path,
            collection_name=self.papers_collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

        chat = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=OPENAI_API_KEY,
        )

        for doi in self.selected_dois:
            print(f"Analyzing [{doi}].")

            article_context = cm.preprocess_doc(
                vector_db,
                doi,
                max_tokens=self.max_paper_tokens,
                literal_queries=self.decision_tree.excerpt_configuration.literal_queries,
                semantic_queries=self.decision_tree.excerpt_configuration.semantic_queries,
            )

            messages, answers, nodes = dtc.evaluate_tree(
                chat,
                self.decision_tree,
                article_context,
                max_reprompts=self.max_reprompts,
            )
            store_results(db, current.run_id, doi, nodes, messages, answers)

            if self.model.startswith("gpt-4") or self.model.endswith("-16k"):
                time.sleep(10)

        self.next(self.end)

    @step
    def end(self):
        """"""
        pass


if __name__ == "__main__":
    AnalyzePapersFlow()

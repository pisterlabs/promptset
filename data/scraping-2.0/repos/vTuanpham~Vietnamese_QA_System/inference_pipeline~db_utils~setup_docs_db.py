import os
import re
import sys
from abc import abstractmethod

sys.path.insert(0, './')
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset

from src.utils import ForceBaseCallMeta, force_super_call

from setup_db import \
    (setup_database, drop_tables, query, insert_data)


def insert_doc(database_path: str,
               max_examples: int=50000):
    ctx_wiki_dataset = load_dataset("EddieChen372/vietnamese-wiki-segmented",
                                    split="train")[:max_examples]
    print(f"Dataset length: {len(ctx_wiki_dataset)}\n")

    def rm_underscore(data: str) -> str:
        return re.sub('_', " ", data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=512 * 0.1,
        length_function=len,
        add_start_index=False,
        separators=["\n\n", "\n", ".", ",", ";", "!", "?", " "],
        keep_separator=True
    )
    texts = text_splitter.create_documents(ctx_wiki_dataset['segmented_text'])
    docs = [rm_underscore(text.page_content) for text in texts]
    data_to_insert = []
    for doc in docs:
        data_to_insert.append({"doc": doc, "source": "EddieChen372/vietnamese-wiki-segmented"})
        print(f"DOC: {doc}\n\n")

    drop_tables("inference_pipeline/dbs/documents.db",
                tables_to_drop=["documents"])
    setup_database("documents",
                   table_names=["documents"],
                   fields=['''(id INTEGER PRIMARY KEY AUTOINCREMENT, doc TEXT, source TEXT)''']
                   )
    insert_data("inference_pipeline/dbs/documents.db",
                table_name='documents',
                data=data_to_insert)

    return docs


if __name__ == "__main__":
    insert_doc("test")

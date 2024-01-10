import random
import json
from langchain.docstore.document import Document
from typing import List, Dict


class DataExtractor:
    """
    A class used to extract a random subset of data from a larger dataset.

    Attributes:
        docs_num (int): The number of documents to randomly extract.
        block_size (int): The size of each block of text to extract from a document.
        blocks (List[str]): The blocks of text extracted from the selected documents.
        output_dir (str): The directory to write the output JSON file.

    Methods:
        extract_blocks_from_documents(docs: List[Document]) -> List[Dict]:
            Extracts a defined block of text from the middle of each selected document.

        extract_to_jsonl_file(docs: List[Document], file_name: str = "eval_blocks.jsonl", need_list: bool = False) -> str:
            Extracts blocks from documents and writes them to a JSON file. Each line in the file represents a block.
    """

    def __init__(self, extract_num=30, block_size=4000, output_dir="Eval"):
        self.blocks = None
        self.docs_num = extract_num
        self.block_size = block_size
        self.output_dir = output_dir

    def extract_blocks_from_documents(self, docs: List[Document]) -> List[Dict]:  # TODO: abstract into extract() and inherit from
        """
        Extracts a defined block of text from the middle of each selected document.
        The number of documents to select is defined by self.docs_num.
        If self.docs_num > len(docs), all documents are selected.
        """
        if self.docs_num > len(docs):
            self.docs_num = len(docs)

        selected_docs = random.sample(docs, self.docs_num)
        blocks = []

        # get text chunks
        for doc in selected_docs:
            content_length = len(doc.page_content)
            start_index = content_length // 2 - self.block_size // 2
            end_index = start_index + self.block_size
            blocks.append(doc.page_content[start_index:end_index])

        return blocks

    def extract_to_jsonl_file(self, docs: List[Document], file_name="eval_blocks.jsonl", need_list=False):
        """
        Extracts blocks from documents and writes them to a JSON file.
        Each line in the file represents a block.
        The file is written to the self.output_dir directory.

        If `need_list` is True, also returns the list of extracted blocks.
        """
        output_path = self.output_dir + '/' + file_name
        with open(output_path, 'w') as f:
            # write text chunks into json file
            block_list = self.extract_blocks_from_documents(docs)
            for i, block in enumerate(block_list, start=1):
                # Construct a dictionary with a single key-value pair, convert to JSON, and write to the file
                line = {
                    "block_id": i,
                    "text": block
                }
                f.write(json.dumps(line) + "\n")

        if need_list:
            return block_list

        return file_name

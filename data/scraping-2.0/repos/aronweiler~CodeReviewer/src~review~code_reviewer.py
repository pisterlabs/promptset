import json
from enum import Enum
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai
from dotenv import dotenv_values
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
import logging
from datetime import datetime
from typing import Union, List, Dict
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from code_reviewer_configuration import CodeReviewerConfiguration
from review.prompts import (
    REVIEW_PROMPT,
    REVIEW_TEMPLATE,
    SUMMARIZE_PROMPT,
    SUMMARIZE_TEMPLATE,
)
from utilities.token_helper import simple_get_tokens_for_message
from review.code_comment import CodeComment
from utilities.open_ai import get_openai_api_key

# TODO: Expand this list- most of the stuff we have doesn't need to be split, anyway.
SUPPORTED_FILE_TYPES = {"py": Language.PYTHON, "cpp": Language.CPP}


class CodeReviewer:
    def __init__(self, configuration: CodeReviewerConfiguration):
        self.configuration = configuration
        self.llm_arguments_configuration = configuration.llm_arguments

        self.remaining_prompt_tokens = (
            self.llm_arguments_configuration.max_supported_tokens
            - self.llm_arguments_configuration.max_completion_tokens
            - simple_get_tokens_for_message(REVIEW_TEMPLATE)
        )

        self.llm = ChatOpenAI(
            model=self.llm_arguments_configuration.model,
            temperature=self.llm_arguments_configuration.temperature,
            openai_api_key=get_openai_api_key(),
            max_tokens=self.llm_arguments_configuration.max_completion_tokens,
        )

        self.review_chain = LLMChain(llm=self.llm, prompt=REVIEW_PROMPT)
        self.summarize_chain = LLMChain(llm=self.llm, prompt=SUMMARIZE_PROMPT)

    def review(self, target_files: List[str]) -> List[CodeComment]:
        # Iterate through the files, and do several things:
        # - Read the file
        # - Split the files into chunks of (max_tokens - max_completion_tokens)
        # - Create embeddings of each chunk so that the LLM can search for them when required
        # - Create an in-memory database of the file chunks and embeddings for later reference by the LLM

        vector_db = self.split_and_add_to_datastore(
            target_files, self.remaining_prompt_tokens
        )

        documents = vector_db.get()
        num_documents = len(documents["documents"])

        logging.info(f"Created vector database with {num_documents} chunks of code")

        comments = []
        # Once the files are split and added to the datastore, we can start the review process
        for i in range(0, num_documents):
            # We don't want to create excessive calls to the LLM, so we will combine chunks into a single call as long as they fit within the remaining prompt tokens

            logging.info(
                f"Reviewing {documents['metadatas'][i]['file_path']}, chunk {documents['metadatas'][i]['chunk']}"
            )
            code_to_review = documents["documents"][i]

            # Summarize the chunk
            if self.configuration.include_summary:
                chunk_summary = self.summarize_chunk(
                    documents["metadatas"][i]["language"], code_to_review
                )
                documents["metadatas"][i]["summary"] = chunk_summary

            # Review the chunk
            chunk_review = self.review_chunk(
                documents["metadatas"][i]["language"], code_to_review
            )
            documents["metadatas"][i]["review"] = chunk_review

            # Load the chunk_review into json
            for comment in self.get_comments(
                chunk_review["text"], documents["metadatas"][i]["file_path"]
            ):
                comments.append(comment)

        return comments

    def get_comments(self, chunk_review: str, file_path: str) -> List[CodeComment]:
        chunk_review_json = json.loads(chunk_review)
        comments = []
        for comment in chunk_review_json["comments"]:
            comments.append(CodeComment(**comment, file_path=file_path))

        return comments

    def split_and_add_to_datastore(
        self, target_files: List[str], max_split_size: int
    ) -> Chroma:
        # Split the file into chunks of (max_tokens - max_completion_tokens)
        # This is because the LLM will need to add the completion tokens to the end of the chunk
        documents = []
        for file in target_files:
            logging.debug(f"Looking at {file}")

            # Get the file extension
            file_extension = file.split(".")[-1]

            # If we support it, continue, otherwise skip it
            if file_extension not in SUPPORTED_FILE_TYPES:
                logging.debug(
                    f"Skipping {file} because it is not a supported file type"
                )
                continue

            language = SUPPORTED_FILE_TYPES[file_extension]
            logging.debug(f"Language is {language} for {file}")

            # Read the file in
            with open(file, "r") as f:
                file_contents = f.read()

            # TODO: When looking at diffs for review, I need to diff two versions of this file,
            # and then extract the chunks of diffs (much like git does),
            # then need to ingest that into the datastore along with the full file
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=max_split_size,
                chunk_overlap=0,
                keep_separator=True,
                add_start_index=True,
                length_function=simple_get_tokens_for_message,
            )

            # Unwind this dumbass list-in-a-list
            joined_docs = code_splitter.create_documents([file_contents])
            docs = [d for d in joined_docs]

            all_lines = file_contents.rstrip().split("\n")

            # Create a list of docs with the medadata
            chunk = 0
            starting_search_line_index = 0
            for d in docs:
                chunk += 1
                # Find the true starting line number
                for line_number, line in enumerate(
                    all_lines[starting_search_line_index:], start=1
                ):
                    if d.page_content.split("\n")[0] in line:
                        starting_line = line_number + starting_search_line_index
                        starting_search_line_index = line_number + 1
                        break

                d.metadata = {
                    "file_path": file,
                    "chunk": chunk,
                    "language": language,
                    "starting_line_number": starting_line,
                }

                # Because of how little LLMs pay attention to things like a single line number prompt,
                # (e.g. ----- BEGIN CODE: Starting Line #: {starting_line_number}  -----)
                # I am going to add line numbers to each line of code
                page_content_lines = d.page_content.split("\n")
                for i, line in enumerate(page_content_lines, start=0):
                    page_content_lines[i] = f"{i + starting_line}: {line}"

                d.page_content = "\n".join(page_content_lines)

                documents.append(d)

            logging.debug(f"Split {file} into {chunk} chunks")

        return Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=get_openai_api_key()))

    def create_embedding(self, text: str, embedding_model="text-embedding-ada-002"):
        return openai.Embedding.create(input=[text], model=embedding_model)["data"][0][
            "embedding"
        ]

    def review_chunk(self, language, code_chunk: str) -> str:
        return self.review_chain(inputs={"language": language, "code": code_chunk})

    def summarize_chunk(self, language, code_chunk: str) -> str:
        return self.summarize_chain(
            inputs={
                "language": language,
                "code": code_chunk,
            }
        )


# Testing
if __name__ == "__main__":
    configuration = CodeReviewerConfiguration.from_environment()
    cr = CodeReviewer(configuration)
    test_json = """{
    "comments": [
        {"start": 2, "end": 2, "comment": "The import statement on line 2 is missing a module name. Please provide the correct module name."},
        {"start": 8, "end": 8, "comment": "The 'constants' module is not imported. Please import it to avoid a NameError on line 8."},
        {"start": 12, "end": 12, "comment": "The default value for 'collection_name' parameter should be 'Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME' instead of 'Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME'. Please update the default value."},
        {"start": 15, "end": 15, "comment": "The 'constants' module is not imported. Please import it to avoid a NameError on line 15."},
        {"start": 26, "end": 26, "comment": "The 'Chroma.from_documents' method is not defined. Please define it or import it from the correct module."},
        {"start": 30, "end": 30, "comment": "The 'constants' module is not imported. Please import it to avoid a NameError on line 30."},
        {"start": 34, "end": 34, "comment": "The 'persist' method is not defined for the 'Chroma' object. Please define it or import it from the correct module."},
        {"start": 35, "end": 35, "comment": "The assignment of 'db' to None on line 35 is unnecessary. The 'db' object will be garbage collected automatically."},
        {"start": 1, "end": 1, "comment": "The 'logging' module is imported but not used. Please remove the import statement if it is not needed."},
        {"start": 3, "end": 3, "comment": "The 'langchain.vectorstores' module is imported but not used. Please remove the import statement if it is not needed."},
        {"start": 6, "end": 10, "comment": "The 'get_chroma_settings' function is not used. Please remove the function if it is not needed."},
        {"start": 20, "end": 20, "comment": "The 'db.get()' method is called without checking if the 'db' object is None. This may result in an AttributeError if 'db' is None."},
        {"start": 33, "end": 33, "comment": "The 'logging' module is imported but not used. Please remove the import statement if it is not needed."}
    ]
}"""

    comments = cr.get_comments(test_json, "testpath.py")

    print(comments)

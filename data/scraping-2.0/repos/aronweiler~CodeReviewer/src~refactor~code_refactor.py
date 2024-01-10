import logging
from typing import List

from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from code_reviewer_configuration import CodeReviewerConfiguration
from refactor.prompts import (
    REFACTOR_PROMPT,
    REFACTOR_TEMPLATE,
    SUMMARIZE_PROMPT,
)
from utilities.token_helper import simple_get_tokens_for_message
from utilities.open_ai import get_openai_api_key

# Supported file types for refactoring
SUPPORTED_FILE_TYPES = {
    "py": Language.PYTHON,
    "cpp": Language.CPP,
}


class CodeRefactor:
    def __init__(self, configuration: CodeReviewerConfiguration):
        self.configuration = configuration
        self.llm_arguments_configuration = configuration.llm_arguments

        # Calculate remaining tokens for prompt
        self.remaining_prompt_tokens = (
            self.llm_arguments_configuration.max_supported_tokens
            - self.llm_arguments_configuration.max_completion_tokens
            - simple_get_tokens_for_message(REFACTOR_TEMPLATE)
        )        

        logging.info(f"Remaining prompt tokens: {self.remaining_prompt_tokens}")

        # If the remaining prompt tokens are lower than this arbitrary number, throw a warning 
        if self.remaining_prompt_tokens < 500:
            logging.warning(
                f"Remaining prompt tokens are low ({self.remaining_prompt_tokens}).  This may cause issues with the refactoring process."
            )

        # Initialize language model
        self.llm = ChatOpenAI(
            model=self.llm_arguments_configuration.model,
            temperature=self.llm_arguments_configuration.temperature,
            openai_api_key=get_openai_api_key(),
            max_tokens=self.llm_arguments_configuration.max_completion_tokens,
        )

        # Initialize refactoring and summarizing chains
        self.refactor_chain = LLMChain(llm=self.llm, prompt=REFACTOR_PROMPT)
        self.summarize_chain = LLMChain(llm=self.llm, prompt=SUMMARIZE_PROMPT)

    def refactor(self, target_files: List[str]):
        # Add target files to datastore and get vector database
        vector_db = self.add_to_datastore(target_files, self.remaining_prompt_tokens)

        documents = vector_db.get()
        num_documents = len(documents["documents"])

        logging.info(f"Created vector database with {num_documents} chunks of code")

        # TODO: Add this step in later
        # First summarize the code's functionality
        # for i in range(0, num_documents):
        #     logging.info(
        #         f"Refactoring {documents['metadatas'][i]['file_path']}"
        #     )
        #     code_to_refactor = documents["documents"][i]

        #     # Summarize the chunk
        #     if self.configuration.include_summary:
        #         chunk_summary = self.summarize_chunk(
        #             code_to_refactor
        #         )
        #         documents["metadatas"][i]["summary"] = chunk_summary

        # Refactor the code        
        last_file_path = ""
        metadata_list = []
        for i in range(num_documents):
            current_file_path = str(documents['metadatas'][i]['file_path'])
            logging.info(f"Refactoring {current_file_path}")
            code_to_refactor = documents["documents"][i]

            # Refactor the chunk
            refactored_code = self.refactor_chain(
                inputs={
                    "code": code_to_refactor,
                    "language": documents["metadatas"][i]["language"],
                }
            )["text"]

            if last_file_path != current_file_path:
                # If it's not the last file name, add new metadata to the list
                metadata_list.append({"file_path": current_file_path, "code": refactored_code})
                # Then update the last file path
                last_file_path = current_file_path
            else:
                # If it is the last file name, append the refactored code to the last metadata in the list
                metadata_list[-1]["code"] += "\n" + refactored_code

        #new_metadata_list = self.get_combined_metadata(metadata_list)
        
        return metadata_list # new_metadata_list    
    
    def get_combined_metadata(self, metadata_list):
        combined_data = {}
        for item in metadata_list:
            file_path = item["file_path"]
            refactored_code = item["code"]
            combined_data.setdefault(file_path, []).append(refactored_code)

        unique_result_list = []
        for file_path, refactored_codes in combined_data.items():
            combined_code = "\n".join(refactored_codes)
            unique_result_list.append({"file_path": file_path, "combined_code": combined_code})

        return unique_result_list

    def add_to_datastore(self, target_files: List[str], max_split_size: int) -> Chroma:
        documents = []
        for file in target_files:
            logging.debug(f"Looking at {file}")

            ## TODO: We don't support files larger than the context window yet, until we get better splitting in place
            if simple_get_tokens_for_message(file) > max_split_size:
                logging.warning(
                    f"Skipping {file} because it is too large to process.  Max size is {max_split_size} tokens.  This will change as soon as I get better splitting in place."
                )
                continue

            # Get the file extension
            file_extension = file.split(".")[-1]

            # If we support it, continue, otherwise skip it
            if file_extension not in SUPPORTED_FILE_TYPES:
                logging.debug(
                    f"Skipping {file} because it is not a supported file type"
                )
                continue

            language = SUPPORTED_FILE_TYPES[file_extension]
            logging.info(f"Language is {language} for {file}")

            # Read the file in
            with open(file, "r") as f:
                file_contents = f.read()

            # TODO: Implement this later, see: https://python.langchain.com/docs/use_cases/code_understanding
            # loader = GenericLoader.from_filesystem(
            #     repo_path+"/libs/langchain/langchain",
            #     glob="**/*",
            #     suffixes=[".py"],
            #     parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
            # )
            # documents = loader.load()

            # Split the code into chunks
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=max_split_size,
                chunk_overlap=0,
                keep_separator=True,
                add_start_index=True,
                length_function=simple_get_tokens_for_message,
            )

            # Create documents from the chunks
            joined_docs = code_splitter.create_documents([file_contents])

            for d in joined_docs:
                d.metadata = {"file_path": file, "language": language}
                logging.debug(f"Adding document {d}")
                documents.append(d)

        logging.info(f"Created {len(documents)} documents, adding to the datastore...")
        return Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=get_openai_api_key()))

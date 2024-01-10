import logging
from typing import List

from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import Blob
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.parsers import LanguageParser
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


from code_reviewer_configuration import CodeReviewerConfiguration
from document.prompts import SUMMARIZE_PROMPT, COMBINE_PROMPT, MAP_REDUCE_COMBINE_PROMPT
from utilities.token_helper import simple_get_tokens_for_message
from utilities.open_ai import get_openai_api_key

# Supported file types for documenting
SUPPORTED_FILE_TYPES = {"py": Language.PYTHON, "cpp": Language.CPP}


class DocumentCode:
    def __init__(self, configuration: CodeReviewerConfiguration):
        self.configuration = configuration
        self.llm_arguments_configuration = configuration.llm_arguments

        # Initialize language model
        self.llm = ChatOpenAI(
            model=self.llm_arguments_configuration.model,
            temperature=self.llm_arguments_configuration.temperature,
            openai_api_key=get_openai_api_key(),
            max_tokens=self.llm_arguments_configuration.max_completion_tokens,
        )

        # Initialize various documentation chains
        self.map_chain = LLMChain(llm=self.llm, prompt=SUMMARIZE_PROMPT)
        self.summarize_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=SUMMARIZE_PROMPT,
            combine_prompt=COMBINE_PROMPT,
            map_reduce_document_variable_name="code",
            combine_document_variable_name="text",
        )

        self.qa_chain = load_qa_chain(llm=self.llm, chain_type="map_reduce", combine_prompt=MAP_REDUCE_COMBINE_PROMPT)

    def document(
        self, target_files: List[str], document_template: str, existing_document: bool
    ):
        # Iterate over the files and load each of them if they are in the supported file types
        documents = []
        for file in target_files:
            file_type = file.split(".")[-1]
            if file_type in SUPPORTED_FILE_TYPES:
                logging.info(f"Processing file: {file}")
                split_docs = self.load_document(file, SUPPORTED_FILE_TYPES[file_type])
                for doc in split_docs:
                    documents.append(doc)
            else:
                logging.warning(f"Skipping unsupported file type: {file_type}")

        datastore = Chroma.from_documents(
            documents, OpenAIEmbeddings(openai_api_key=get_openai_api_key())
        )

        retrieval_qa = RetrievalQA(
            combine_documents_chain=self.qa_chain,
            retriever=datastore.as_retriever(
                # Fetch more documents for the MMR algorithm to consider, but only return the top 10
                search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}
            ),
        )
        # summary = self.summarize_chain.run(documents)

        if existing_document:
            # If we are updating existing documentation, we need to do the following:
            # 1. Load the existing documentation
            # 2. Split by sections / or at least paragraphs and sentences
            # 3. Analyze the existing chunks of documentation, including the purpose of the various chunks
            # 4. Do "fact checking" on the existing documentation using the loaded documents in the data store
            # 5. Add or update the existing documentation with what we find
            raise NotImplementedError(
                "Updating existing documentation is not yet implemented"
            )
        else:
            logging.info(f"Loading document template from: {document_template}")

            # This will just return a list of the "[Prompt] ..." strings in the template
            with open(document_template, "r") as f:
                template_contents = f.read()

            prompt_strings = self.get_template_prompts(template_contents)

            # Iterate over the prompts, and answer them using the loaded documents
            for prompt in prompt_strings:
                logging.info(f"Prompt: {prompt}")

                result = retrieval_qa.run(prompt.lower().replace("[prompt]", "").strip())

                # Replace the prompt in the template with the result
                template_contents = template_contents.replace(prompt, result)

            return template_contents


    def get_template_prompts(self, template_contents: str):
        

        # Split on newline
        lines = template_contents.split("\n")

        # Find all of the lines that start with "[Prompt]"
        prompt_strings = [line for line in lines if line.startswith("[Prompt]")]
        logging.info(f"Found {len(prompt_strings)} prompts in the template")

        return prompt_strings

    def load_document(self, file_path: str, language: Language):
        # Load the file
        # with open(file_path, "r") as f:
        #     file_contents = f.read()

        parser = LanguageParser(language=language)

        documents = parser.parse(Blob(path=file_path))

        # Split the code into chunks
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=500,
            chunk_overlap=0,
            keep_separator=True,
            add_start_index=True,
            length_function=simple_get_tokens_for_message,
        )

        # Create documents from the chunks
        split_documents = code_splitter.split_documents(documents)

        return split_documents

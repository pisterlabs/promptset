import os
import re
import shutil
from pathlib import Path
from typing import List

from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, GenerationConfig





class CogninovaSearch:
    """This class is responsible for searching the knowledge base and generating answers to user queries"""

    def __init__(self, cfg_model, cfg_search, cfg_vs, llm, embedding):
        """
        :param cfg_model: The model configuration
        :param cfg_search: The search configuration
        :param cfg_vs: The vector storage configuration
        :param llm: The language model
        :param embedding: The embedding object
        """
        self.cfg_model = cfg_model
        self.cfg_search = cfg_search
        self.cfg_vs = cfg_vs
        self.search_type = self.cfg_search.get("search_type")
        self.k = self.cfg_search.getint("k_return")
        self.chain_type = self.cfg_search.get("chain_type")
        self.persist_dir = self.cfg_vs.get("persist_dir")
        self.vdb_type = self.cfg_vs.get("vdb_type")

        self.gen_config = GenerationConfig(
            temperature=self.cfg_model.getfloat("temperature"),
            top_k=self.cfg_model.getint("top_k"),
            top_p=self.cfg_model.getfloat("top_p"),
            num_beams=self.cfg_model.getint("num_beams"),
            max_new_tokens=self.cfg_model.getint("max_new_tokens"),
        )
        self.llm = llm
        self.embedding = embedding
        self.vector_db = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg_model.get("name"), device_map="auto")
        
        # create a file debug.txt to store the debug messages. the file will be overwritten each time the app is run.
        # later in the code, we will write to this file using self.f.write("message")
        self.f = open("debug.txt", "w")

        self._load_document()
        self._load_vector_database()

    def _load_document(self) -> None:
        """
        Load documents from a directory and create embeddings for them
        """
        loaded_docs = []
        document_dir = self.cfg_vs.get("docs_dir")
        if isinstance(document_dir, str):
            document_dir = Path(document_dir)

        loaded_docs_dir = document_dir / ".loaded_docs/"
        loaded_docs_dir.mkdir(exist_ok=True)
        count_new_files_loaded = 0
        for file in document_dir.iterdir():
            is_new_file = not (loaded_docs_dir / file.name).exists()

            if not is_new_file: print(f"Skipping {file.name} since it is already loaded")

            if file.suffix == ".pdf" and is_new_file:
                print(f"Loading {file.name}")
                loader = PyPDFLoader(str(file))
                data_txt = loader.load()
                loaded_docs.extend(data_txt)

                shutil.copy(str(file), str(loaded_docs_dir / file.name))  # Copy the file to the loaded_docs_dir
                count_new_files_loaded += 1

        if count_new_files_loaded > 0:
            print(f"Loaded {count_new_files_loaded} new files. Creating embeddings...")

            self._store_embeddings(loaded_docs)
            print(f"Created embeddings for {count_new_files_loaded} new files.")

    def _store_embeddings(self, loaded_docs):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.cfg_vs.getint("chunk_size"),
            chunk_overlap=self.cfg_vs.getint("chunk_overlap")
        )
        splits = text_splitter.split_documents(loaded_docs)

        if self.vdb_type == "chroma":
            # TODO: From here, search for professional alternatives (cloud based vector databases ?)
            self.vector_db = Chroma.from_documents(
                documents=splits, embedding=self.embedding, persist_directory=self.persist_dir
            )
            self.vector_db.persist()
        else:
            raise NotImplementedError(f"Vector database type {self.vdb_type} not implemented")

    def _load_vector_database(self) -> None:
        """
        Load the vector database from the persist directory
        """
        if self.vdb_type == "chroma":
            self.vector_db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding)
        else:
            raise NotImplementedError(f"Vector database type {self.vdb_type} not implemented")

    def search(self, query, filter_on=None) -> List:
        """
        :param query: The query to search for (input from the user in natural language)
        :param filter_on: If set, filter the search on the document. The filter is a dictionary with one key. It can be
        either "source" or "page". (i.e. {"source":"docs/cs229_lectures/Lecture03.pdf"} or {"page": "1"})
        """
        assert self.search_type in ["similarity", "mmr"], \
            f"search_type must in ['similarity', 'mmr'] got {self.search_type}"


        if self.search_type == "similarity":
            result = self.vector_db.similarity_search(query, k=self.k, filter=filter_on)
        else:  # mmr
            result = self.vector_db.max_marginal_relevance_search(query, k=self.k, filter=filter_on)

        return result

    def answer(self, query, search_result, template_obj=None) -> str:
        """
        :param query: The query to search for (input from the user in natural language)
        :param search_result: Result of the search using "similarity" or "mmr" in self.search()
        :param template_obj: The CogninovaTemplate object
        :return: The answer to the query
        """
        assert template_obj is not None, "retrieval_template_obj must be provided"
        assert self.chain_type in ["refine", "stuff"], f"chain_type must in ['refine', 'stuff'] got {self.chain_type}"

        guess = ""

        self.f.write(
            "<h2 style='background-color: #404854; padding:10px; border-radius:5px; margin-bottom:3px;'>"
            f"⛓️ Chain type: {self.chain_type}"
            "</h2>"
        )

        if self.chain_type == "stuff":
            document_separator = "\n\n"
            context = []
            for res in search_result:
                chunked_content = res.page_content
                context.append(chunked_content)

            context_str = document_separator.join(context)
            prompt_template = PromptTemplate(
                template=template_obj.stuff_template, input_variables=["context", "question"])
            prompt = prompt_template.format(context=context_str, question=query)
            guess = self.run_inference(prompt)

            guess_alpha_num = re.sub(r'\W+', '', guess)
            if guess_alpha_num.strip() == "" or len(guess_alpha_num) <= 1:
                guess = "I don't know."

            self.f.write("<div style='background-color: #5F6B7C; padding:10px; border-radius:5px;'>")
            self.f.write("<strong style='color: #1C2127;'>⌨️ Retrieval Prompt</strong><br>")
            self.f.write(f"<em><strong>{prompt}</strong></em>")
            self.f.write(f"<p style='color:#EC9A3C';><strong>{guess}</strong></p>")
            self.f.write("</div>")


        elif self.chain_type == "refine":
            # First guess
            first_context = search_result[0].page_content
            inputs = ["context", "question"]
            prompt_template = PromptTemplate(template=template_obj.refine_template_start, input_variables=inputs)
            prompt = prompt_template.format(context=first_context, question=query)
            guess = self.run_inference(prompt)

            guess_alpha_num = re.sub(r'\W+', '', guess)
            if guess_alpha_num.strip() == "" or len(guess_alpha_num) <= 1:
                guess = "I don't know."

            old_guess = guess

            self.f.write("<div style='background-color: #5F6B7C; padding:10px; border-radius:5px;'>")
            self.f.write("<strong style='color: #1C2127;'>⌨️ Retrieval Prompt n°1</strong><br>")
            self.f.write(f"<em><strong>{prompt}</strong></em>")
            self.f.write(f"<p style='color:#EC9A3C';><strong>{guess}</strong></p>")
            self.f.write("</div>")

            # Refine the answer
            other_contexts = search_result[1:]

            if len(other_contexts) > 0:
                for n, next_context in enumerate(other_contexts):
                    next_context = next_context.page_content
                    inputs = ["question", "guess", "context"]
                    prompt_template = PromptTemplate(template=template_obj.refine_template_next, input_variables=inputs)
                    prompt = prompt_template.format(context=next_context, question=query, guess=guess)
                    guess = self.run_inference(prompt)

                    guess_alpha_num = re.sub(r'\W+', '', guess)
                    if guess_alpha_num.strip() == "" or len(guess_alpha_num) <= 1:
                        guess = old_guess

                    self.f.write("<div style='background-color: #5F6B7C; padding:10px; border-radius:5px;'>")
                    self.f.write(f"<strong style='color: #1C2127;'>⌨️ Retrieval Prompt n°{n+2}</strong><br>")
                    self.f.write(f"<em><strong>{prompt}</strong></em>")
                    self.f.write(f"<p style='color:#EC9A3C';><strong>{guess}</strong></p>")
                    self.f.write("</div>")


                self.f.write("<div style='background-color: #5F6B7C; padding:10px; border-radius:5px;'>")
                self.f.write(f"<p style='color:#EC9A3C';><strong>Final Answer: {guess}</strong></p>")
                self.f.write("</div>")

        self.f.flush()
        return guess

    def run_inference(self, prompt) -> str:
        """
        Run inference on the prompt
        :param prompt: The user query
        :return: The answer to the query
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        model_output = self.llm.generate(input_ids=input_ids, generation_config=self.gen_config)
        response = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        return response


    @staticmethod
    def reset_persist_directory(persist_dir) -> None:
        """
        Delete the persist directory
        :param persist_dir: The directory to delete
        :return: None
        """
        if not isinstance(persist_dir, str):
            persist_dir = str(persist_dir)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

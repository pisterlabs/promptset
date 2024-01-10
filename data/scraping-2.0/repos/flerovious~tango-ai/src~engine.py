from pathlib import Path
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.query_engine import CitationQueryEngine
from prompt_templates import guidance_prompt_tmpl
from enums import Expertise


class TangoEngine:
    def __init__(self, expertise: Expertise):
        self.expertise = expertise.value
        self.load_dotenv()
        self.project_root = self.find_project_root() / "src"
        self.data_dir = self.project_root / "data" / self.expertise
        self.storage_dir = self.project_root / "storage" / self.expertise
        self.index = self.load_or_create_index()
        self.query_engine = self.create_query_engine()

    def find_project_root(self):
        # Starting from the current file's directory
        current_dir = Path(__file__).parent
        # Traverse up until you find the project root marker (e.g., a '.git' directory)
        while not (current_dir / ".git").exists() and current_dir != current_dir.parent:
            current_dir = current_dir.parent
        return current_dir

    def load_dotenv(self):
        load_dotenv()

    def load_or_create_index(self):
        if not self.storage_dir.exists():
            return self.create_index()
        else:
            return self.load_index_from_storage()

    def create_index(self):
        documents = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            required_exts=[".pdf"],
            recursive=True,
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(
            persist_dir=str(self.storage_dir),
        )
        return index

    def load_index_from_storage(self):
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.storage_dir)
        )
        return load_index_from_storage(storage_context)

    def create_query_engine(self):
        query_engine = CitationQueryEngine.from_args(
            self.index, similarity_top_k=3, citation_chunk_size=512
        )
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": guidance_prompt_tmpl}
        )
        return query_engine

    def query(self, question, options, answer):
        response = self.query_engine.query(
            f"""
            Question: 
            {question}

            Options:
            {options}

            Answer:
            {answer}
            """
        )
        return response

    def print_response(self, response):
        for node in response.source_nodes:
            page_label = node.node.metadata["page_label"]
            file_name = node.node.metadata["file_name"]
            print(f"Page {page_label}")
            print(f"File {file_name}")
            print(
                f"Url https://github.com/gabyang/tango-ai/tree/main/src/data/14/{file_name.replace(' ', '%20')}"
            )


if __name__ == "__main__":
    expertise_engine = TangoEngine("chemistry")
    question = "What are some important conjugate acid-base pair related to foods?"
    options = """A) NaCl
B) HCl
C) HCl0
D) H2SO4"""
    answer = "C) HCl0"
    response = expertise_engine.query(question, options, answer)
    expertise_engine.print_response(response)

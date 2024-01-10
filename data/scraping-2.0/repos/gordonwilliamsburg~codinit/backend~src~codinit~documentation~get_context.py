import logging
import os
import re
from typing import List, Optional

import weaviate
from apify_client import ApifyClient
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

from codinit.config import (
    DocumentationSettings,
    Secrets,
    documentation_settings,
    secrets,
)
from codinit.documentation.chunk_documents import chunk_document
from codinit.documentation.doc_schema import documentation_file_class, library_class
from codinit.documentation.pydantic_models import Library, WebScrapingData
from codinit.documentation.save_document import load_scraped_data_from_json
from codinit.weaviate_client import get_weaviate_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BaseWeaviateDocClient:
    """
    Base class for weaviate Documentation client.
    """

    def __init__(self, library: Library, client: weaviate.Client) -> None:
        self.library = library
        self.client = client
        self.init_schema()

    def check_library_exists(self):
        # query if library already exists and has documentation files
        result = (
            self.client.query.get(
                "Library",
                properties=["name"],
            )
            .with_where(
                {
                    "path": ["name"],
                    "operator": "Equal",
                    "valueText": self.library.libname,
                }
            )
            .do()
        )
        library_exists = result["data"]["Get"]["Library"]
        if len(library_exists) == 0:
            return False
        else:
            return True

    def get_lib_id(self) -> Optional[str]:
        object_id = None
        result = (
            self.client.query.get(
                "Library",
                properties=["name"],
            )
            .with_where(
                {
                    "path": ["name"],
                    "operator": "Equal",
                    "valueText": self.library.libname,
                }
            )
            .with_additional(properties=["id"])
            .do()
        )
        # get the id of a library from weaviate
        if (
            "data" in result
            and "Get" in result["data"]
            and "Library" in result["data"]["Get"]
            and len(result["data"]["Get"]["Library"])
            > 0  # case where library exists at least once
        ):
            if len(result["data"]["Get"]["Library"]) > 1:
                logging.warning(
                    f"More than one library with name {self.library.libname} found, returning first one."
                )
            object_id = result["data"]["Get"]["Library"][0]["_additional"]["id"]
        else:  # case where result is {'data': {'Get': {'Library': []}}}
            logging.warning(f"Library ID for {self.library.libname} not found.")
        return object_id

    def check_library_has_docs(self, lib_id: str) -> int:
        """returns number of documents associated with one library in the database"""
        num_docs = 0
        result = (
            self.client.query.get(
                "Library",
                properties=[
                    "name",
                    "hasDocumentationFile {... on DocumentationFile {title}}",
                ],
            )
            .with_where({"path": ["id"], "operator": "Equal", "valueString": lib_id})
            .do()
        )
        logging.debug(
            f"Querying documentation for library with ID {lib_id} gives {result=}"
        )
        # get the number of documentation files associated with a library"
        """
        1. example, no docs exist:  result = {'data': {'Get': {'Library': [{'hasDocumentationFile': None,
            'name': 'langchain'}]}}}
        2. example, multiple docs exist: result = {'data': {'Get': {'Library': [{'hasDocumentationFile': [{'title': 'title1'}, {'title': 'title2'}], 'name': 'langchain'}]}}}
        """
        # Check if there are associated documentation files
        if (
            "data" in result
            and "Get" in result["data"]
            and "Library" in result["data"]["Get"]
        ):
            library_data = result["data"]["Get"]["Library"]
            logging.debug(
                f"library with ID {lib_id} has documentation files: {library_data=}"
            )
            """
            1. example, no docs exist: library_data = [{'hasDocumentationFile': None, 'name': 'langchain'}]
            2. example, multiple docs exist: library_data = [{'hasDocumentationFile': [{'title': 'title1'}, {'title': 'title2'}], 'name': 'langchain'}]
            """
            if (
                library_data
                and library_data[0]
                and ("hasDocumentationFile" in library_data[0])
            ):
                documentation_files = library_data[0]["hasDocumentationFile"]
                if documentation_files:
                    num_docs = len(documentation_files)
                else:
                    logging.warning(
                        f"Library with ID {lib_id} has no documentation files."
                    )
            else:  # case where result={'data': {'Get': {'Library': []}}}
                logging.error(f"Error getting library with ID {lib_id} from weaviate.")
        return num_docs

    # TODO create test for this class
    def init_schema(self):
        existing_schema = self.client.schema.get()
        existing_classes = {cls["class"] for cls in existing_schema.get("classes", [])}

        required_classes = {library_class["class"], documentation_file_class["class"]}

        # Check if all required classes already exist
        if not required_classes.issubset(existing_classes):
            # Create schema with both classes
            self.client.schema.create(
                {"classes": [library_class, documentation_file_class]}
            )
            logging.info(
                f"Created schema with classes {library_class['class']} and {documentation_file_class['class']}"
            )


# refactor the following document to put all functions under one class
class WeaviateDocLoader(BaseWeaviateDocClient):
    """
    loads the documentation of a library and save it to weaviate.
    """

    def __init__(
        self,
        library: Library,
        client: weaviate.Client,
        documentation_settings: DocumentationSettings = documentation_settings,
        secrets: Secrets = secrets,
    ):
        # superinit class
        super().__init__(library=library, client=client)
        self.documentation_settings = documentation_settings
        self.secrets = secrets
        self.apify_client = ApifyClient(secrets.apify_key)

    def load_json(self, filename: str) -> List[WebScrapingData]:
        return load_scraped_data_from_json(filename=filename)

    def get_raw_documentation(self) -> List[WebScrapingData]:
        """
        get the raw documentation from json file
        """
        data = []
        docs_dir = self.secrets.docs_dir
        filename = docs_dir + "/" + self.library.libname + ".json"
        # TODO check if all urls are present in the json file
        if os.path.exists(filename):
            logging.info(
                f"Loading scraped Documentation for library {self.library.libname} from {filename=}"
            )
            # load data using load_scraped_data_from_json function from codinit.documentation.save_document
            data = self.load_json(filename=filename)
        else:
            logging.error(f"{filename=} does not exist.")
        return data

    def chunk_doc(self, doc: WebScrapingData) -> List[str]:
        # chunk document using chunk_document function from codinit.chunk_documents.py
        chunks = chunk_document(
            document=doc.text,
            chunk_size=self.documentation_settings.chunk_size,
            overlap=self.documentation_settings.overlap,
        )
        return chunks

    # save document to weaviate
    def save_doc_to_weaviate(self, doc_obj: dict, lib_id: str) -> str:
        # TODO create hash of an object and query against object hash in library. If not found then save object.
        doc_id = self.client.data_object.create(
            data_object=doc_obj, class_name="DocumentationFile"
        )
        # DocumentationFile -> Library relationship
        self.client.data_object.reference.add(
            from_class_name="DocumentationFile",
            from_uuid=doc_id,
            from_property_name="fromLibrary",
            to_class_name="Library",
            to_uuid=lib_id,
        )

        # Library -> DocumentationFile relationship
        self.client.data_object.reference.add(
            from_class_name="Library",
            from_uuid=lib_id,
            from_property_name="hasDocumentationFile",
            to_class_name="DocumentationFile",
            to_uuid=doc_id,
        )
        logging.info(
            f"Saved document with DOC_ID {doc_id=} to library with LIB_ID {lib_id=}"
        )
        return doc_id

    # save library to weaviate

    def save_lib_to_weaviate(self):
        # create library object
        lib_obj = {
            "name": self.library.libname,
            "links": self.library.links,
            "description": self.library.lib_desc,
        }
        lib_id = self.client.data_object.create(
            data_object=lib_obj, class_name="Library"
        )
        logging.info(
            f"Saved library object {lib_id=} to weaviate,library has LIB_ID {lib_id=}"
        )
        return lib_id

    def get_or_create_library(self):
        """
        get or create library in weaviate
        returns lib_id
        """
        # check if library already exists
        if self.check_library_exists():
            # get library id
            lib_id = self.get_lib_id()
        else:
            # create library
            lib_id = self.save_lib_to_weaviate()
        return lib_id

    def get_or_create_documentation(self, doc_obj: dict, lib_id: str):
        """
        get or create documentation in weaviate
        returns doc_id
        """
        doc_id = self.save_doc_to_weaviate(doc_obj=doc_obj, lib_id=lib_id)
        return doc_id

    # embed documentation to weaviate
    def embed_documentation(self, data: List[WebScrapingData], lib_id: str):
        """Chunk documents and load them to weaviate.

        Args:
            data (List[WebScrapingData]): list of WebScrapingData objects
        """
        # iterate over data
        for doc in data:
            # chunk document using chunk_document function from codinit.chunk_documents.py
            chunks = self.chunk_doc(doc=doc)
            # iterate over chunks, with chunk and its order in doc
            for chunk_num, chunk in enumerate(chunks):
                # create doc_obj of the chunk according to DocumentationFile schema
                doc_obj = {
                    "title": doc.metadata.title,
                    "description": doc.metadata.description,
                    "chunknumber": chunk_num,
                    "source": str(doc.url),
                    "language": doc.metadata.languageCode,
                    "content": chunk,
                }
                # save chunk to weaviate
                doc_id = self.get_or_create_documentation(
                    doc_obj=doc_obj, lib_id=lib_id
                )
                logging.info(
                    f"Processed loading for chunk document with DOC_ID {doc_id=}"
                )

    # run

    def run(self):
        # get or create library
        lib_id = self.get_or_create_library()
        data = self.get_raw_documentation()
        if len(data) == 0:
            print("No data found.")
        else:
            # embed documentation to weaviate
            self.embed_documentation(data=data, lib_id=lib_id)


class WeaviateDocQuerier(BaseWeaviateDocClient):
    """
    queries the documentation of a library from weaviate.
    """

    def __init__(
        self,
        library: Library,
        client: weaviate.Client,
        documentation_settings: DocumentationSettings = documentation_settings,
    ) -> None:
        super().__init__(library=library, client=client)
        self.documentation_settings = documentation_settings
        # create retriever
        self.retriever = self.get_retriever()

    # get retriever
    def get_retriever(self):
        # create retriever
        retriever = WeaviateHybridSearchRetriever(
            client=self.client,
            index_name="DocumentationFile",
            text_key="content",
            k=self.documentation_settings.top_k,
            alpha=self.documentation_settings.alpha,
        )
        return retriever

    # get relevant documents for a query
    def get_relevant_documents(self, query: str):
        # clean up query that might be produced by an LLM
        query = query.replace("`", "").replace("'", "").replace('"', "")
        result = re.findall(r'"(.*?)"', query)
        if len(result) > 0:
            query = result[0]
        print(query)
        docs = self.retriever.get_relevant_documents(query=query)
        # print(f"{docs=}")
        relevant_docs = ""
        for doc in docs:
            relevant_docs += doc.page_content
        return relevant_docs


if __name__ == "__main__":
    libname = "langchain"
    links = [
        "https://langchain-langchain.vercel.app/docs/get_started/",
        # "https://python.langchain.com/docs/modules/",
        # "https://python.langchain.com/docs/use_cases",
        # "https://python.langchain.com/docs/guides",
        # "https://python.langchain.com/docs/integrations",
    ]
    client = get_weaviate_client()
    library = Library(libname=libname, links=links)
    weaviate_doc_loader = WeaviateDocLoader(library=library, client=client)
    weaviate_doc_loader.run()
    """
    weaviate_doc_querier = WeaviateDocQuerier(
        library=library, client=weaviate_doc_loader.client
    )
    docs = weaviate_doc_querier.get_relevant_documents(
        query="Using the langchain library, write code that illustrates usage of the library."
    )
    print(docs)
    print(weaviate_doc_loader.get_lib_id())
    num_docs = weaviate_doc_loader.check_library_has_docs(lib_id="some_id")
    print(num_docs)
    """

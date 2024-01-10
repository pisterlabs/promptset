import json
import os.path
from pathlib import Path
from string import Template
from typing import List, Iterator, Iterable, Optional, cast, Mapping, Callable

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document, BaseDocumentTransformer
from langchain.text_splitter import TextSplitter

from eurelis_kb_framework.base_factory import JSON


class TemplateTextDocWrapper:
    def __init__(self, document: Document):
        self.document = document

    def __getitem__(self, key: str):
        if key == "page_content":
            return self.document.page_content
        elif key.startswith("meta_"):
            return self.document.metadata.get(key[5:], "")

        raise ValueError(f"Unhandled key {key} in text_template")


class Dataset(BaseLoader):
    """
    Define a datasource (Dataloader) associated with an optional transformer and splitter.
    """

    def __init__(self, dataset_id: str, loader: BaseLoader):
        """
        Dataset
        Args:
            dataset_id: id for the dataset
            loader: langchain document_loader to use
        """
        self.id = dataset_id
        self.loader = loader
        self.splitter = None
        self.transformer = None
        self.output_folder = None
        self.output_file_varname = "id"
        self.index = True
        self.cleanup = None
        self.source_id_key = "source"
        self.name = dataset_id
        self.vector_store = None
        self.embeddings = None
        self.metadata = None
        self.text_template: Optional[Template] = None

    def set_text_template(self, value: str):
        """Setter for the text_template
        Args:
            value (str): the value
        """
        self.text_template = Template(value)

    def has_template(self) -> bool:
        return bool(self.text_template)

    def apply_text_template(self, doc: Document):
        substitute_dict = cast(Mapping[str, object], TemplateTextDocWrapper(doc))
        doc.page_content = self.text_template.safe_substitute(substitute_dict)

    def set_splitter(self, splitter: TextSplitter):
        """
        Setter for the splitter
        Args:
            splitter: a TextSplitter instance

        Returns:

        """
        self.splitter = splitter

    def set_transformer(self, transformer: BaseDocumentTransformer):
        """
        Setter for the transformer
        Args:
            transformer: a BaseDocumentTransformer instance

        Returns:

        """
        self.transformer = transformer

    def set_output_folder(self, folder: str):
        """
        Setter for output_folder variable

        Args:
            folder: where to write documents to

        Returns:

        """
        self.output_folder = folder

    def set_source_id_key(self, source_id_key: str):
        """
        Setter for the source id key parameter
        Args:
            source_id_key: source id key parameter, used for indexing

        Returns:

        """
        self.source_id_key = source_id_key

    def set_output_file_varname(self, varname: str):
        """
        Setter for output_file_varname variable

        Args:
            varname: key of the metadata value to use as filename for cache output

        Returns:

        """
        self.output_file_varname = varname

    def set_metadata(self, metadata: dict):
        """
        Setter for metadata variable
        Args:
            metadata:

        Returns:

        """
        self.metadata = metadata

    def set_index(self, index: JSON):
        if index is None:
            return

        if isinstance(index, bool) and not index:
            self.index = False
            return

        if isinstance(index, str):
            self.index = index
            if self.index != "cache":
                raise ValueError(
                    f"Invalid 'index' parameter value in {self.id} dataset"
                )
            return

        if not isinstance(index, dict):
            raise ValueError(f"Invalid 'index' parameter value in {self.id} dataset")

        self.source_id_key = index.get("source_id_key", "source")
        self.name = index.get("name", self.id)
        self.cleanup = index.get("cleanup", None)
        if not self.cleanup:
            self.cleanup = None

        if self.cleanup is not None and self.cleanup not in {"full", "incremental"}:
            raise ValueError(
                f"Invalid 'index.cleanup' parameter in dataset {self.id}, should be either false, not set, full or incremental"
            )

    @staticmethod
    def load_document_from_cache(path: str) -> Document:
        """
        Helper method to load a document from a path
        Args:
            path: path of a json document cache file

        Returns:
            langchain document object

        """
        with open(path) as json_file:
            doc_json = json.load(json_file)
            page_content = doc_json.get("page_content")
            metadata = doc_json.get("metadata")

            return Document(page_content=page_content, metadata=metadata)

    def _write_document_as_cache(self, document: Document):
        """
        Helper method to write a document to the disk as a json file
        Args:
            document: langchain document to write

        Returns:

        """
        json_doc = {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }

        relative_path = document.metadata.get(self.output_file_varname)

        if self.splitter:
            relative_path += f"-{document.metadata.get('start_index', 0)}"

        output_folder = Path(self.output_folder)

        cache_path = Path(os.path.join(self.output_folder, relative_path + ".json"))
        file_folder = Path(os.path.dirname(cache_path))

        if (
            output_folder not in cache_path.parents
        ):  # ensure we are still in the output folder
            raise RuntimeError(
                f"Path for cache file {str(cache_path)} is not inside output folder {str(output_folder)}"
            )

        os.makedirs(file_folder, exist_ok=True)  # create sub folders if needed

        with open(cache_path, "w") as json_file:
            json.dump(json_doc, json_file)

    def _lazy_load_transformer(self) -> Iterator[Document]:
        """
        Helper method to get an iterator over transformed documents
        Returns:
            iterator over transformed documents
        """

        # first we get documents from the loader
        try:
            documents = self.loader.lazy_load()
        except NotImplementedError:
            documents = self.loader.load()

        if not self.transformer and not self.metadata:  # if no transformer was defined
            yield from documents
        elif not self.transformer:
            for doc in documents:
                doc.metadata.update(self.metadata)
                yield doc
        elif self.metadata:
            for doc in documents:  # for each document we proceed to transformation
                doc.metadata.update(self.metadata)
                yield from self.transformer.transform_documents([doc])
        else:
            for doc in documents:  # for each document we proceed to transformation
                yield from self.transformer.transform_documents([doc])

    def _lazy_load_splitter(self) -> Iterator[Document]:
        """
        Helper method to get an iterator over splitted documents
        Returns:
            iterator over splitted documents
        """
        documents = self._lazy_load_transformer()

        if not self.splitter:  # No splitter was defined
            yield from documents

        else:
            for doc in documents:
                yield from self.splitter.split_documents([doc])

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """
        Preferred method to get documents, using a lazy loader
        Returns:
            iterator over already splitted documents

        """
        return self._lazy_load_splitter()

    # Sub-classes should implement this method
    # as return list(self.lazy_load()).
    # This method returns a List which is materialized in memory.
    def load(self) -> List[Document]:
        """
        Legacy method to load documents as a list
        Returns:
            list of documents
        """
        return list(self.lazy_load())

    def write_files(self, output):
        """
        Method to write cache files
        Args:
            output: object to print to the console

        Returns:

        """

        def do_work():
            for document in self.lazy_load():
                self._write_document_as_cache(document)

        output.status(f"Writing cache files for '{self.id}' dataset", do_work)

    def get_first_doc(self) -> Optional[Document]:
        """
        Helper method to get the first document of the dataset
        Returns:
            first document of the dataset

        """
        try:
            return next(self.lazy_load())
        except StopIteration:
            return None

    @staticmethod
    def print_datasets(output, datasets: Iterable["Dataset"], verbose_only=False):
        """
        Method to print datasets to an output
        Args:
            output: output object to use
            datasets: list of datasets
            verbose_only: default to False, if True use the output verbose variant

        Returns:

        """
        console_print_table = (
            output.verbose_print_table if verbose_only else output.print_table
        )

        console_print_table(
            datasets,
            ["ID", "Can index?", "Can cache?"],
            lambda index, dataset: (
                dataset.id,
                str(dataset.index),
                str(bool(dataset.output_folder)),
            ),
            title="Datasets",
        )

    def set_embeddings(self, embeddings):
        """
        Setter for embeddings
        Args:
            embeddings: embeddings object

        Returns:

        """
        self.embeddings = embeddings

    def set_vector_store(self, vector_store):
        """
        Setter for vector store
        Args:
            vector_store: vector store object

        Returns:

        """
        self.vector_store = vector_store

    def build_with_namespace_function(
        self, project: str
    ) -> Callable[[Iterator[Document]], Iterator[Document]]:
        """Build the "with_namespace" function used before indexing

        Args:
            project (str): name of the project

        Returns:
            a method taking documents in entry and returning documents
        """

        def with_namespace(documents: Iterator[Document]) -> Iterator[Document]:
            # two different loops for performance reason
            if self.has_template():
                for document in documents:
                    document.metadata["namespace"] = f"{project}/{self.name}"
                    # in this case we apply the text template
                    self.apply_text_template(document)
                    yield document
            else:
                for document in documents:
                    document.metadata["namespace"] = f"{project}/{self.name}"
                    yield document

        return with_namespace

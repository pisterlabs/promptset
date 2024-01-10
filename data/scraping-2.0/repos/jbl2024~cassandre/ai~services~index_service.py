import logging
import os
import re
import tempfile

from django.conf import settings
from django.core.files.storage import default_storage
from langchain.document_loaders import PDFPlumberLoader, TextLoader
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import SpacyTextSplitter
from ai.services.hints import parse_hints

from documents.models import Category, Correction, Document

from . import vector_database_service
from .chunk import split_markdown

logger = logging.getLogger("cassandre")


def clean_text(text, full=True):
    """
    Cleans the input text based on the specified rules.

    Args:
        text (str): The text to be cleaned.
        full (bool, optional): If True, applies full cleaning including removal of
        duplicated consecutive end of lines, leading and trailing spaces,
        and duplicated consecutive spaces. Defaults to True.

    Returns:
        str: The cleaned text.
    """
    # Remove duplicated consecutive spaces
    if full is True:
        # Remove duplicated consecutive end of lines
        text = re.sub(r"\n+", "\n", text)

        # Remove leading and trailing spaces
        text = text.strip()

        text = re.sub(r"\s+", " ", text)
    # Special case for FAQ documents where each question has a "réponse" block
    # so we add separator so that the splitter will keep question/answer together
    text = text.replace("Question:", "\n\n\n\nQuestion:")

    return text


def get_categories(category_id=None):
    """
    Fetches categories based on the provided category_id.

    Args:
        category_id (int, optional): The id of the category to be fetched.
        If None, fetches all categories. Defaults to None.

    Returns:
        QuerySet: A QuerySet containing the fetched categories.
    """
    return (
        Category.objects.all()
        if category_id is None
        else Category.objects.filter(id=category_id)
    )


def load_and_split_pdf(temp_file, document):
    """
    Load and split a PDF document.

    Args:
        temp_file (File): The temporary file containing the PDF document.
        document (str): The name of the document.

    Returns:
        list: A list of split text documents.
    """
    loader = PDFPlumberLoader(temp_file.name)
    loaded_documents = process_loaded_documents(loader.load(), document)

    text_splitter = SpacyTextSplitter(
        pipeline="fr_core_news_lg",
        chunk_size=settings.SPLIT_CHUNK_SIZE,
        chunk_overlap=settings.SPLIT_CHUNK_OVERLAP,
    )
    splitted_documents = text_splitter.split_documents(loaded_documents)

    hint_mappings = parse_hints(getattr(document, "hints", ""))

    for doc in splitted_documents:
        page_number = doc.metadata.get("page", 1)
        specific_hints = hint_mappings.get(page_number, [])
        all_hints = hint_mappings.get("all", [])

        # Combine all hints applicable to this page
        combined_hints = specific_hints + all_hints
        if combined_hints:
            doc.page_content = f"{' - '.join(combined_hints)}\n" + doc.page_content
        else:
            doc.page_content = f"{doc.metadata.get('origin', '')}\n" + doc.page_content
    return splitted_documents


def load_and_split_md(temp_file, document):
    """
    Load and split a markdown file into smaller documents.

    Args:
        temp_file (File): A temporary file object containing the markdown content.
        document (str): The name of the document.

    Returns:
        list: A list of split documents.

    Raises:
        None

    Notes:
        - This function uses a TextLoader to load the content from the temporary file.
        - The loaded content is then processed using process_loaded_documents().
        - The processed documents are split using a MarkdownTextSplitter.

    """
    loader = TextLoader(temp_file.name)
    loaded_documents = process_loaded_documents(
        loader.load(), document, full_clean=False
    )
    splitted_docs = []
    for doc in loaded_documents:
        items = split_markdown(doc.page_content)
        for item in items:
            splitted_docs.append(
                LangchainDocument(page_content=item, metadata=doc.metadata)
            )

    return splitted_docs


def process_loaded_documents(loaded_documents, document, full_clean=True):
    """
    Process the loaded documents.

    Args:
        loaded_documents (list): A list of documents to be processed.
        document (Document): The document object.

    Returns:
        list: The processed list of documents.
    """
    for doc in loaded_documents:
        doc.page_content = clean_text(doc.page_content, full_clean)
        doc.metadata["origin"] = document.title or os.path.basename(document.file.name)
        # fix page number starting at 0
        # see https://github.com/langchain-ai/langchain/pull/10653
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1

    return loaded_documents


def index_documents(category_id=None):
    """
    Indexes all the documents in the specified category.

    :param category_id: The ID of the category to index documents for.
    If not specified, all categories will be indexed.
    """
    categories = get_categories(category_id)

    for category in categories:
        documents = Document.objects.filter(category=category)
        texts = []

        for document in documents:
            with default_storage.open(document.file.name, "rb") as file:
                file_content = file.read()
                file_ext = os.path.splitext(document.file.name)[-1]

                with tempfile.NamedTemporaryFile(
                    suffix=file_ext, delete=False
                ) as temp_file:
                    logger.info("Loading: %s", document.title or document.file)
                    temp_file.write(file_content)
                    temp_file.flush()

                    if file_ext == ".pdf":
                        texts.extend(load_and_split_pdf(temp_file, document))
                    elif file_ext == ".md":
                        texts.extend(load_and_split_md(temp_file, document))

                    logger.info(
                        "Successfully loaded document: %s",
                        document.title or document.file,
                    )

        corrections = Correction.objects.filter(category=category)
        for correction in corrections:
            document = LangchainDocument(
                page_content=f"Question: {correction.query}\nRéponse: {correction.answer}",
                metadata={
                    "origin": "correction manuelle",
                    "source": f"correction-{correction.id}",
                    "page": 1,
                },
            )
            texts.append(document)

        vector_database_service.create_collection(category.slug, texts)

        logger.info(
            "Successfully indexed %s document(s) and %s correction(s) for category %s",
            documents.count(),
            corrections.count(),
            category.name,
        )

    logger.info("Successfully indexed all documents")

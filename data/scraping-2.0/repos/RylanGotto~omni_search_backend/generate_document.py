import re
from langchain.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from unstructured.cleaners.core import clean, clean_extra_whitespace
import pprint
from unstructured.documents.html import HTMLDocument


class GenerateDocument:
    @staticmethod
    def generate(url):
        document = []

        loader = UnstructuredURLLoader(
            urls=[url],
            mode="elements",
            post_processors=[clean, clean_extra_whitespace],
        )
        elements = loader.load()
        # print(elements)
        # exit()
        selected_elements = [
            e
            for e in elements
            if e.metadata["category"] == "ListItem"
            or e.metadata["category"] == "NarrativeText"
        ]
        full_clean = re.sub(
            " +", " ", " ".join([e.page_content for e in selected_elements])
        )

        return full_clean

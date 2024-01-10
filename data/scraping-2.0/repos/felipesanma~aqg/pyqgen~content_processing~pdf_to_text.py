from langchain.document_loaders import UnstructuredFileLoader
import re


class PDF2text:
    def __init__(self) -> None:
        pass

    def group_content_per_page(self, *, documents: list) -> dict:
        content_per_page = {}
        for doc in documents:
            if doc.metadata["page_number"] in content_per_page:
                content_per_page[doc.metadata["page_number"]] += f" {doc.page_content}"
            else:
                content_per_page[doc.metadata["page_number"]] = doc.page_content
        return content_per_page

    def clean_text(self, *, s: str) -> str:
        clean = re.compile("<.*?>")
        s = re.sub(clean, "", s)
        s = s.replace("\r", " ")
        s = re.sub(r"\.+", ".", s)
        # s = s.replace("\n", " ").replace("\r", " ")
        # s = s.replace(":selected:", "").replace(":unselected:", "")
        # s = s.replace('\"', '')
        # s = s.replace(".", "")
        return s

    def extract(self, *, pdf_file):
        loader = UnstructuredFileLoader(pdf_file, strategy="fast", mode="elements")
        documents = loader.load()

        pdf_pages_content = "\n".join(doc.page_content for doc in documents)
        clean_content = self.clean_text(s=pdf_pages_content)

        content_per_page = self.group_content_per_page(documents=documents)
        for k, v in content_per_page.items():
            content_per_page[k] = self.clean_text(s=v)

        return clean_content, content_per_page

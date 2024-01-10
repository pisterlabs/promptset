from langchain.document_loaders import PDFMinerPDFasHTMLLoader, PDFMinerLoader
from langchain.schema.document import Document

class Custom_Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def read_as_html(pdf_file_path: str) -> Document:
    try:
        # Using PDFminer, we are extracting the PDF as an HTML file
        # This process retains the necessary formatting info of the original PDF
        # Now we can use features like checking font size, style etc
        loader = PDFMinerPDFasHTMLLoader(pdf_file_path)

        # loader expects multiple PDFs; since we are using 1 PDF, we want the first item of the list.
        # The first item is stored at the 0th index
        data = loader.load()[0]
        return data
    except Exception as e:
        # Handle any exceptions that may occur during the PDF loading process
        # Create a Custom_Document instance with predefined values
        page_content = '<html>/html>\n'
        metadata = {'source': pdf_file_path}
        return Custom_Document(page_content, metadata)

def read_as_text(pdf_path: str) -> str:
    try:
        loader = PDFMinerLoader(pdf_path)
        data = loader.load()
        text = ""
        for page in data:
            text += page.page_content
        return text
    except Exception as e:
        # Handle any exceptions that may occur during the PDF loading process
        # Return an empty string or handle the error as needed
        return ""

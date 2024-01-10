from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfquery import PDFQuery


def start(filename):
    if is_csv_file(filename):
        text = "not implemented yet"
    if is_pdf_file(filename):
        text = read_text_from_pdf(filename)
    return split_text_to_chunks(text)


# main function for splitting text into chunks
# you can customize chunk_size and chunk_overlap
def split_text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20
    )
    chunks = text_splitter.create_documents([text])
    print(len(chunks))
    for chunk in chunks:
        print(chunk)
    return chunks


def read_text_from_pdf(filename):
    pdf = PDFQuery(filename)
    pdf.load()

    # Use CSS-like selectors to locate the elements
    text_elements = pdf.pq('LTTextLineHorizontal').text()

    return text_elements


def is_csv_file(filename):
    return parse_extension(filename) == "csv"


def is_pdf_file(filename):
    return parse_extension(filename) == "pdf"


def parse_extension(filename):
    return filename.split(".")[1]


start('NASDAQ_AAPL_2019.pdf')

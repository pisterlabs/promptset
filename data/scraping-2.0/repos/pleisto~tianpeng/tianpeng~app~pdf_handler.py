import PyPDF2
from tianpeng.app import pg_vector_util
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector


def extract_text(filepath):
    # Open the PDF file in read-binary mode
    with open(filepath, "rb") as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Create an empty string to store the text
        text = ""

        # Loop through each page in the PDF file
        for page_num in range(len(pdf_reader.pages)):
            # Update the progress bar
            print("progress: " + str(page_num + 1) + "/" + str(len(pdf_reader.pages)))

            # Get the page object
            page_obj = pdf_reader.pages[page_num]

            # Extract the text from the page
            page_text = page_obj.extract_text()

            # Add the text to the string
            text += page_text

    return text


def split_text(text):
    text_splitter = CharacterTextSplitter()
    return text_splitter.split_text(text)


def write_textstr_to_db(text, collection):
    # CONNECTION_STRING = pg_vector_util.pg_conn()
    CONNECTION_STRING = pg_vector_util.get_conn_string()
    txts = split_text(text)
    embeddings = OpenAIEmbeddings()

    db = PGVector.from_texts(
        embedding=embeddings,
        texts=txts,
        collection_name=collection,
        connection_string=CONNECTION_STRING,
    )
    db.create_vector_extension()
    # query = "What is frax"
    # docs_with_score = db.similarity_search_with_score(query)
    # for doc, score in docs_with_score:
    #     print("-" * 80)
    #     print("Score: ", score)
    #     print(doc.page_content)
    #     print("-" * 80)


# if __name__ == "__main__":
#     text = extract_text("image/frax.pdf")
#     print(text)

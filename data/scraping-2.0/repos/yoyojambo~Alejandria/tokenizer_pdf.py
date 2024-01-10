from langchain.vectorstores import Chroma

from langchain.document_loaders import PyPDFLoader

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# loader = PyPDFLoader("pdfs/economia.pdf")
# loader = PyPDFLoader("pdfs/astronomy.pdf")



content_list = []

# files = ["1. Que es la estrategia.pdf", "2. Recursos & Capacidades dinámicas.pdf", "2. Ventaja centrada en Recursos y Capacidades.pdf", "3. Estrategias_disruptivas_Cadena Valor.pdf", "Factors Críticos de Exito (FCE).pdf", "M1-T1.pdf", "M1-T2.pdf", "M1-T3.pdf", "Que es Estrategia.pdf"]

files = ["astronomy.pdf"]

for file in files:
    p = "pdfs/" + file
    loader = PyPDFLoader(p)
    content_list += loader.load_and_split()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
)


toked = splitter.transform_documents(content_list)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(toked, embedding_function, persist_directory="./memory")


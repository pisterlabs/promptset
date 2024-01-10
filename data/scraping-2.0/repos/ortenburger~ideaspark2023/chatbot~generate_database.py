import chromadb
from chromadb.config import Settings as ChromaSettings

import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import openai_config
import chroma_config

pdf_urls=[
    (
        'https://www.sparkassenversicherung.de/export/sites/svag/_resources/download_galerien/bedingungen/23-708.pdf',
        'ask_sv_police_haftpflicht'
    ),
    (
        'https://www.oerag.de/export/sites/oerag/_resources/downloads/produkte/ARB_2023.pdf',
        'ask_sv_police_rechtsschutz'
    ),
    (
        'https://www.sparkassenversicherung.de/export/sites/svag/_resources/download_galerien/bedingungen/43-014.pdf',
        'ask_sv_police_sorglos-leben'
    ),
    (
        'https://mein-premiumservice.vkb.de/service2/rest/api/material-service/public/v1/material/335597',
        'ask_sv_police_auslandsreisekrankenversicherung'
    ),
    (
        'https://mein-premiumservice.vkb.de/service2/rest/api/material-service/public/v1/material/335591',
        'ask_sv_police_auslandsreisekrankenversicherung'
    ),
    (
        'https://www.sparkassenversicherung.de/export/sites/svag/_resources/download_galerien/bedingungen/23-681.pdf',
        'ask_sv_police_hausratversicherung'
    ),
    (
        'https://www.sparkassenversicherung.de/export/sites/svag/_resources/download_galerien/bedingungen/23-600.pdf',
        'ask_sv_police_internetschutz'
    ),
    (
        'https://www.s-mobilgeraeteschutz.de/online/avb/AVB_S-Mobilgeräteschutz_GAVB-SMG-1G-07-22_v120722.pdf',
        'ask_sv_police_mobilgeraeteschutz'
    ),
    (
        'https://www.sparkassenversicherung.de/export/sites/svag/_resources/download_galerien/bedingungen/23-713.pdf',
        'ask_sv_police_tierhalterhaftpflicht'
    )
]

chroma_settings=ChromaSettings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",chroma_client_auth_credentials=chroma_config.CHROMA_TOKEN)
chroma_client=chromadb.HttpClient(host=chroma_config.CHROMA_HOST, port=chroma_config.CHROMA_PORT, settings=chroma_settings)
chroma_client.reset()
embeddings = OpenAIEmbeddings(deployment="policeembedding")

for pdf_url, collection_name in pdf_urls:
  loader = PyPDFLoader(pdf_url)
  data = loader.load_and_split()

  # Token Text Splitter alternative
  text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=10, chunk_overlap=0)
  documents = text_splitter.split_documents(data)

  db=Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embeddings)

  try:
    db.add_documents(documents)
  except:
    db.add_documents(documents)

  sleep(1)



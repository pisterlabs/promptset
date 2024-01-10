"""Diese Datei dient dazu, das Sprachmodell für die Stufe 1 zu initialisieren."""

import os
import openai
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def get_chain(path):
    """Nimmt einen Pfad zu einer ifc-Datei und gibt eine Chain zurück, welche Fragen zur ifc-Datei beantworten kann."""

    # Setzt den OpenAI API Key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Lädt das IFC-Modell mithilfe von einem TextLoader ein
    loader = TextLoader(path)
    ifc_model = loader.load()

    # Teilt das Modell zur besseren Verarbeitung in kleinere Texte auf
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    texts = text_splitter.split_documents(ifc_model)

    # Importiert die Embedding Funktion von OpenAI
    embeddings = OpenAIEmbeddings()

    # Erstellt die Embeddings für die einzelnen Texte und speichert diese in einer Datenbank (Vectorstore)
    docsearch = Chroma.from_documents(texts, embeddings)

    # Prompt template für das Sprachmodell
    prompt_template = """Given the following ifc file that contains information about a building information model with filetype
                ifc and a question, create a final answer.
                Answer detailed and in full sentences. Round all number to two decimal places.
                If you don't know the answer, dont try to come up with an answer. Just write "I don't know".
                Below the short answer you should add a long answer that contains more information and reasons for your answer (Max. 2 sentences).

                Respond in German.
                
                Model name: {context}

                QUESTION: {question}
                =========
                FINAL ANSWER IN German:"""

    # Erstellt ein Prompt mithilfe des Templates
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
    chain_type_kwargs = {"prompt": prompt}

    # Konfiguriert das Sprachmodell von OpenAI und wählt gpt-3.5-turbo-16k aus
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)

    # Erstellt die Chain mit dem Sprachmodell, dem Prompt und den gespeicherten Embeddings
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)

    # Gibt die Chain zurück
    return chain


def generateAnswer(path, query, chain):
    """Nimmt einen Pfad zu einer IFC-Datei, eine Frage und eine Chain und gibt eine Antwort auf die Frage zurück."""

    # Speichert den Namen des Modells als Kontext für das Modell
    context = path.split("/")[-1].split(".")[0]

    # Erstellt eine Antwort mithilfe der Chain, der Frage und dem Kontext
    response = chain({"query": query, "context": context})

    # Gibt die Antwort zurück
    return response['result']


if __name__ == '__main__':
    """Testet die Funktionen der Datei."""

    path = "../data_ifc_models/Beispielhaus.ifc"
    query = "Wie viele Brandwände weisen eine Feuerwiederstandklasse von F60 auf?"
    chain = get_chain(path)
    print(f"{query}:")
    print(generateAnswer(path, query, chain))

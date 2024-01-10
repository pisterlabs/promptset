from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import re
import pickle

os.environ["OPENAI_API_KEY"] = "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"


def load_code_files(directory):
    code_text = ''
    for root, dirs, files in os.walk(directory):
        if "third-party" in root:
            continue
        for file in files:
            if file.endswith(('.cpp', '.h')):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    code_text += f.read() + '\n'
    return code_text

code_directory = '../../src'
texte_brut = load_code_files(code_directory)
# exit(0)
separateur = "\n"
decoupeur_texte = CharacterTextSplitter(
    separator=separateur,
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
textes = decoupeur_texte.split_text(texte_brut)

modeles_embedding = OpenAIEmbeddings()
docsearch = FAISS.from_texts(textes, modeles_embedding)
chain = load_qa_chain(OpenAI(), chain_type="stuff")
file = open("stored_data.pickle", "wb")
pickle.dump({
    "chain": chain,
    "docsearch": docsearch
}, file)
file.close()

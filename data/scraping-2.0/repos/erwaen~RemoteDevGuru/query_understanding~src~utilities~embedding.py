from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from openai.embeddings_utils import get_embedding, cosine_similarity
import os
import pandas as pd

def embedding(self, doc_name: str = "remote-work-paraguay", chunk_size: int = 300) :
        """Script para poder generar los archivo picke con el embeding de cada parrafo"""
        path_name = f"{self.doc_dir}/{doc_name}"
        path_file = f"{path_name}.pdf"

        if not os.path.isfile(path=path_file):
            raise FileNotFoundError()
        
        loader = PyPDFLoader(path_file)
        # dividimos en paginas 
        pages = loader.load_and_split()

        # Un elemento por cada página
        # return pages[3].page_content
        
        # Divide las paginas en parrafos de {chunk_size} caracteres
        # en caso de que identifique el salto de linea junto con un punto final, tambien lo separa
        split = CharacterTextSplitter(chunk_size=chunk_size, separator = '.\n')
        
        # ejecuta la funcion que divide las paginas en parrafos
        textos = split.split_documents(pages) # Lista de textos
        # print(textos[4].page_content)
        # Extraemos la parte de page_content de cada texto y lo pasamos a un dataframe
        textos = [str(i.page_content) for i in textos] #Lista de parrafos
        parrafos = pd.DataFrame(textos, columns=["texto"])
        # print(parrafos)

        parrafos['Embedding'] = parrafos["texto"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002')) # Nueva columna con los embeddings de los parrafos
        
        # parrafos.to_csv(f'{self.doc_dir}REMOTE.csv')
        parrafos.to_pickle(f'{path_name}-emb.pk')

        return parrafos
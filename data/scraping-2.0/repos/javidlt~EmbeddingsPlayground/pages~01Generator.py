import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import operator
import cohere
import numpy as np

st.set_page_config(
    page_title="Generador",
    page_icon="👋",
)

st.write("# Generar embeddings")

def generate(mod, df, colText):
    embedder = SentenceTransformer(mod)
    tot = len(df)
    dfW = df
    dfW["Embedding"] = None
    progress_text = "Generando embeddings"
    my_bar = st.progress(0, text=progress_text)
    for index, row in dfW.iterrows():
        pro = int(((index+1)/tot)*100)
        embedding = embedder.encode(str(row[colText])).tolist()
        dfW.at[index, 'Embedding'] = embedding
        my_bar.progress(pro, text=progress_text)
    return dfW

def generateCohere(mod, df, colText, apiKey):
    co = cohere.Client(apiKey)
    doc_emb = co.embed(df[colText].astype(str).tolist(), input_type="search_document", model=mod).embeddings
    doc_emb = np.asarray(doc_emb)
    return doc_emb

def convert_to_json(df):
    jsonToRet = pd.DataFrame.from_dict(df)
    return jsonToRet.to_json(index=False)

def convert_to_csv(df):
    csvToRet = pd.DataFrame.from_dict(df)
    return csvToRet.to_csv(index=False)

if 'listOfFilesNamesGenerate' not in st.session_state:
    st.session_state.listOfFilesNamesGenerate = []
if 'listOfDictsGenerateEmbd' not in st.session_state:
    st.session_state.listOfDictsGenerateEmbd = []
if 'indexOfDataset' not in st.session_state:
    st.session_state.indexOfDataset = 0
if 'uploaded_file_count' not in st.session_state:
    st.session_state.uploaded_file_count = 0
if 'dfWithGeneratedEmbeddings' not in st.session_state:
    st.session_state.dfWithGeneratedEmbeddings = {}
if 'datasetToUseGen' not in st.session_state:
    st.session_state.datasetToUseGen = ""

uploaded_fileCount = st.session_state.uploaded_file_count
datasetToUse = st.session_state.datasetToUseGen

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "excel", "json"])
if uploaded_file is not None and (uploaded_file.name not in st.session_state.listOfFilesNamesGenerate):
    if st.sidebar.button('usar archivo'):
        uploaded_fileCount = uploaded_fileCount+1

if uploaded_file is not None and (uploaded_fileCount != st.session_state.uploaded_file_count):
    # Can be used wherever a "file-like" object is accepted:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    dictEmbd = df.to_dict()
    st.session_state.listOfDictsGenerateEmbd.append(dictEmbd)
    st.session_state.listOfFilesNamesGenerate.append(uploaded_file.name)
    st.session_state.uploaded_file_count = st.session_state.uploaded_file_count+1

if st.session_state.listOfDictsGenerateEmbd != []:
    st.session_state.datasetToUseGen = st.sidebar.radio("Dataset a usar", st.session_state.listOfFilesNamesGenerate)
    st.session_state.indexOfDataset = st.session_state.listOfFilesNamesGenerate.index(st.session_state.datasetToUseGen)
    dfEmbd = pd.DataFrame.from_dict(st.session_state.listOfDictsGenerateEmbd[st.session_state.indexOfDataset])
    column_names = list(dfEmbd.columns.values)
    st.session_state.columnGenWiText = st.selectbox('Nombre de columna con texto', column_names)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
           st.session_state.typeGen = st.radio("Modelo para embeddings",["**default**", "**Cualquier modelo huggingFace**"],)
        with col2:
            if st.session_state.typeGen == "**default**":
                st.session_state.modelGen = st.selectbox(
                    'Modelo',
                    ('ggrn/e5-small-v2', 'Cohere/Cohere-embed-english-v3.0', 'Cohere/Cohere-embed-multilingual-v3.0', 'intfloat/multilingual-e5-small', 'intfloat/e5-small-v2', 'sentence-transformers/all-MiniLM-L6-v2'))
            else: 
                st.session_state.modelGen = st.text_input('Modelo')
    
    if 'Cohere' in st.session_state.modelGen:
        st.session_state.CohereAPIGenerate = st.text_input('API KEY')

    dfFinal = pd.DataFrame()
    if st.button('Generar embeddings', type="primary"):
        if 'Cohere' in st.session_state.modelGen:
            dfFinal = generateCohere(st.session_state.modelGen, dfEmbd,st.session_state.columnGenWiText, st.session_state.CohereAPIGenerate)
        else:
            print("si entra")
            dfFinal = generate(st.session_state.modelGen, dfEmbd,st.session_state.columnGenWiText)
            print(dfFinal)
            st.session_state.dfWithGeneratedEmbeddings = dfFinal.to_dict()

    if st.session_state.dfWithGeneratedEmbeddings != {}:
        json = convert_to_json(st.session_state.dfWithGeneratedEmbeddings)
        csv = convert_to_csv(st.session_state.dfWithGeneratedEmbeddings)
        with st.container ():
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Descargar json",
                    json,
                    f"{st.session_state.listOfFilesNamesGenerate[st.session_state.indexOfDataset]}_Embeddings.json",
                    "text/json",
                    key='download-json'
                )
            with col2:
                st.download_button(
                    "Descargar csv",
                    csv,
                    f"{st.session_state.listOfFilesNamesGenerate[st.session_state.indexOfDataset]}_Embeddings.csv",
                    "text/csv",
                    key='download-csv'
                )
        dfToPrint = pd.DataFrame.from_dict(st.session_state.dfWithGeneratedEmbeddings)
        if datasetToUse != st.session_state.datasetToUseGen:
            st.markdown("**Se ha cambiado el dataset con el que estas trabajando, descarga el resultado o se borrará tu avance cuando des click a generar.**")
        st.write(dfToPrint)
else:
    st.markdown(
    """
    ### Pasos 
    - Subir json, csv o excel con la columna de texto con la que deseas generar los embeddings
    - Escribir cuál es la columna del texto
    - Seleccionar el modelo con el que se harán los embeddings
    - Exportar json con tus embeddings para usarlo
    """
    )
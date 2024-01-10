import openai
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Configuração do Streamlit
st.set_page_config(
    page_title="Chatbot com OpenAI",
    page_icon=":robot:",
    layout="wide",
)

# Sidebar para inserir a chave da API da OpenAI
st.sidebar.header("Chave da API da OpenAI")
chave = st.sidebar.text_input("Chave da API", type="password")

# Verificação se a chave da API foi fornecida
if chave:
    openai.api_key = chave

    # Título da página
    st.title("Chatbot com OpenAI")

    # Função para usar a API da OpenAI
    def Perguntar(prompt, persona):
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            stop=None,
            messages=[{"role": "system", "content": persona},
                      {"role": "user", "content": prompt}]
        )
        return completion['choices'][0]['message']['content']

    # Upload de arquivo PDF
    uploaded_file = st.file_uploader("Carregar um arquivo PDF", type=["pdf"])

    if uploaded_file is not None:
        st.text(
            "Lendo o PDF... Isso pode levar até 2 minutos para um documento de teste.")
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Lê o arquivo PDF
        loader = PyPDFLoader("temp_pdf.pdf")
        # Organiza em pedaços com o número da página
        pages = loader.load_and_split()

        st.text(
            "Criando o VectorStore... Isso pode levar até 40 segundos para um documento de teste.")
        faiss_index = FAISS.from_documents(
            pages, OpenAIEmbeddings(model="text-embedding-ada-002"))

        st.text("Terminou de ler o PDF.\nFazendo a requisição à API.")

        pergunta = st.text_input("Faça uma pergunta:")

        if st.button("Perguntar"):
            if pergunta:
                bibliografia = ""
                termo_pesquisa = ""

                # Procurando os 3 termos com maior proximidade com a pergunta.
                docs = faiss_index.similarity_search(pergunta, k=3)

                # Pega o conteúdo guardado
                for doc in docs:
                    termo_pesquisa += "Conteúdo: " + \
                        str(doc.page_content) + "\n\n"
                    bibliografia += f"Página: {doc.metadata['page']} \nConteúdo:\n{doc.page_content} \n\n"

                # Prepara a pergunta
                termo_pesquisa += f"Use este conteúdo acima se for relevante para a pergunta abaixo:\n{pergunta}"

                # Melhora a personalidade
                persona = "Você é um especialista no assunto da pergunta. Deve responder de forma clara, pensando cuidadosamente antes de escrever."

                # Envia a requisição
                final = Perguntar(termo_pesquisa, persona)
                resposta = final

                # Exibir resposta e bibliografia
                st.text("Resposta:")
                st.write(resposta)

                st.text("Referências:")
                st.write(bibliografia)
else:
    st.warning(
        "Por favor, insira sua Chave da API da OpenAI na barra lateral para continuar.")

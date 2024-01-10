import os
import pandas as pd
import streamlit as st
import base64
import time
from datetime import datetime
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import OpenAIModerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
#from langchain.chains import RetrievalQA
import pinecone
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class CustomModeration(OpenAIModerationChain):
    def __init__(self, openai_api_key):
        super().__init__(openai_api_key=openai_api_key)

    def _moderate(self, text: str, results: dict) -> str:
        if results["flagged"]:
            error_str = f'Moderação: A mensagem "{text}" não é apropriada. Tente novamente com outra mensagem.'
            return error_str
        return ""

def nivel_criatividade(x):
    dicionario = {
        'Conservador': 0.0,
        'Equilibrado': 0.5,
        'Criativo': 0.8,
        'Inovador': 1.2,
        'Experimental': 1.6,
        'Livre / Sem Limites': 2.0
    }
    return dicionario.get(x)

def estilo_resposta(x):
    dicionario = {
        'Infantil': 'Um estilo de resposta lúdico e simples, próprio para crianças, com frases curtas, divertidas e muitos emojis.',
        'Minimalista': 'Um estilo de resposta sucinto e direto, focado apenas nos pontos principais.',
        'Extrovertido': 'Um estilo mais expressivo e vibrante, com uso de linguagem mais informal e envolvente.',
        'Erudito': 'Um estilo de resposta refinado e formal, utilizando uma linguagem sofisticada e estruturas complexas.',
        'Artístico': 'Um estilo criativo e poético, com uso de metáforas e linguagem figurativa.',
        'Técnico': 'Um estilo mais detalhado e preciso, utilizando terminologias específicas.'
    }
    return dicionario.get(x)

def tamanho_resposta(x):
    dicionario = {
        'Pequeno': 50,
        'Médio': 100,
        'Grande': 150
    }
    return dicionario.get(x)

def finalizar(tokens_utilizados, custo_estimado, c0, index_name):

    # Cria um dataframe do Pandas para armazenar os detalhes da conversa
    dados_conversa = pd.DataFrame(columns=["Dia e Hora", "Tokens Utilizados", "Custo Estimado", "Histórico do Diálogo"])

    # Obtém informações relevantes
    dia_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    historico_formatado = []

    # Construindo o histórico formatado
    for mensagem in st.session_state.mensagens:
        historico_formatado.append(f"{mensagem['role']}: {mensagem['content']}")

    # Juntando as mensagens formatadas em uma única string
    historico_com_quebras_de_linha = "\n".join(historico_formatado)

    dados_conversa.loc[0] = [dia_hora, tokens_utilizados, round(custo_estimado,5), historico_formatado]

    texto_conteudo = '\n'.join([f"{col}: {row[col]}\n" for _, row in dados_conversa.iterrows() for col in dados_conversa.columns])

    # Salva o histórico da conversa em um arquivo PDF
    pdf_file_name = "historico_conversa.pdf"
    
    doc = SimpleDocTemplate(pdf_file_name, pagesize=letter)
    story = []

    # Define um estilo para o texto
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']

    story.append(Paragraph("Histórico do Diálogo", styles['Title']))
    story.append(Paragraph(f"Dia e Hora: {dia_hora}", style_normal))
    story.append(Paragraph(f"Tokens Utilizados: {tokens_utilizados}", style_normal))
    story.append(Paragraph(f"Custo Estimado: {round(custo_estimado,5)}", style_normal))
    story.append(Paragraph("Histórico do Diálogo:", style_normal))
    
    for mensagem in historico_formatado:
        story.append(Paragraph(mensagem, style_normal))


    # Constrói o PDF
    doc.build(story)


    # Disponibiliza um link para download do arquivo PDF gerado
    #st.markdown(f"### [Baixar histórico da conversa como PDF]({pdf_file_name})")
    with open(pdf_file_name, "rb") as pdf_file:
        pdf_base64 = base64.b64encode(pdf_file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="{pdf_file_name}">Baixar Histórico da Conversa em PDF</a>'
    c0.markdown(href, unsafe_allow_html=True)
    
    if index_name is not None and index_name != "":
        # Verifica se o objeto Pinecone foi inicializado
        if 'pinecone' in locals() or 'pinecone' in globals():
            if index_name in pinecone.list_indexes():
                pinecone.delete_index(index_name)
    
    st.session_state.mensagens = [{"role": 'system', "content": 'Você será um amigo para conversar sobre história do Brasil!'}]
    st.empty()


st.set_page_config(
    page_title="Assistente Virtual",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded")

st.header("💬 Assistente Virtual")

#openai_api_key = st.sidebar.text_input('Chave da API OpenAI', type = 'password')
#pinecone_key = st.sidebar.text_input('Chave do Pinecone', type = 'password')
openai_api_key = st.secrets["my_secret"]["OPENAI_API_KEY"]
pinecone_key = st.secrets["my_secret"]["PINECONE_API_KEY"]
if 'pdf_file_name' not in st.session_state:
    st.session_state.pdf_file_name = None
tokens_utilizados = 0
custo_estimado = 0

uploaded_file = st.sidebar.file_uploader("Selecione seus PDFs aqui", type=['pdf'])

col1, col2, col3 = st.columns(3)

# Montagem dos 3 filtros da tela
with col1:
    criatividade = st.select_slider(
        label='Nível de Criatividade',
        options=['Conservador', 'Equilibrado', 'Criativo', 'Inovador', 'Experimental', 'Livre / Sem Limites'],
        value='Conservador',
        format_func=lambda x: x,
        help='Slider de Seleção'
    )

    # Use o valor selecionado para definir a temperatura
    temperatura = nivel_criatividade(criatividade)
    st.write(f"Criatividade selecionada: {temperatura}")

with col2:
    # Seleção do tamanho da resposta em tokens
    tamanho = st.select_slider(
        label='Tamanho da Resposta',
        options=['Pequeno', 'Médio', 'Grande'],
        value='Pequeno',
        format_func=lambda x: x,
        help='Slider de Seleção'
    )

    # Use o valor selecionado para definir a quantidade de tokens
    quantidade_tokens = tamanho_resposta(tamanho)
    st.write(f"Quantidade de Tokens selecionada: {quantidade_tokens}")

with col3:
    # Seleção do estilo da resposta
    estilo = st.selectbox(
        label='Estilo da Resposta',
        options=['Infantil','Minimalista', 'Extrovertido', 'Erudito', 'Artístico', 'Técnico'],
        index=2,
        format_func=lambda x: x,
        help='Selecione o estilo de resposta desejado'
    )

    # Use o valor selecionado para definir o estilo da resposta
    estilo_selecionado = estilo_resposta(estilo)
    st.write(f"{estilo_selecionado}")  





#llm = ChatOpenAI(
#    openai_api_key=openai_api_key,
#    model_name='gpt-3.5-turbo',
#    temperature=temperatura
#)

# Iniciar Historico Chat
if "mensagens" not in st.session_state:
    st.session_state.mensagens = [{"role": 'system', "content": 'Você será um amigo para conversar sobre história do Brasil!'}] 


# Aparecer o Historico do Chat na tela
for mensagens in st.session_state.mensagens[1:]:
    with st.chat_message(mensagens["role"]):
        st.markdown(mensagens["content"])

# Campo para o usuário digitar o seu prompt
prompt = st.chat_input("Digite alguma coisa")

with st.spinner("Processando"):
    if openai_api_key:
        llm = OpenAI(api_key=openai_api_key,temperature=temperatura)
        index_name = ""
        if uploaded_file is not None and prompt:
            if pinecone_key:
                with open("file.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                    file_name = uploaded_file.name
                # Carregar o arquivo PDF
                loader = PyPDFLoader("file.pdf")
                pages = loader.load_and_split()
                os.remove("file.pdf")
                
                if st.session_state.pdf_file_name != file_name:
                    
                    st.session_state.pdf_file_name = file_name

                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

                    pinecone.init(api_key=pinecone_key,
                                  environment="gcp-starter")
                    index_name = "vector-database"
                    if index_name not in pinecone.list_indexes():
                        # we create a new index
                        pinecone.create_index(
                            name=index_name,
                            metric='cosine',
                            dimension=1536  # 1536 dim of text-embedding-ada-002
                        )
                        st.session_state.pine = Pinecone.from_documents(pages, embeddings, index_name = index_name)
                        # wait a moment for the index to be fully initialized
                        time.sleep(20) 
                    else:
                        st.session_state.pine = Pinecone.from_documents(pages, embeddings, index_name = index_name)
            
                busca = st.session_state.pine.similarity_search(prompt, k = 3)
                conteudo_pagina = []
                num_resultados = len(busca)

                if num_resultados > 0:
                    for i in range(num_resultados):
                        page_content = busca[i].page_content.replace('\n', ' ')
                        conteudo_pagina.append(page_content)
                else:
                    st.warning('Nenhum resultado encontrado no arquivo ou não houve tempo suficiente para indexar o arquivo.\nA pergunta será respondida sem levar o arquivo em consideração, mas você pode tentar perguntar novamente depois.')

                #qa = RetrievalQA.from_chain_type(
                #    llm=llm,
                #    chain_type="stuff",
                #    retriever=pine.as_retriever()
                #)

                prompt_final = f"""
                Responda com um estilo {estilo}: {estilo_selecionado}
                Responda com no máximo {tamanho} tokens sem perder o sentido da resposta.
                Elabore bem as respostas mas não fuja do tema da pergunta do usuário.

                Responda a pergunta do usuário considerando o conhecimento a seguir.

                """

                # Adicionando o conteúdo das páginas ao prompt final
                for i, content in enumerate(conteudo_pagina):
                    prompt_final += f"{content}\n\n"

                # Adicionando informações adicionais ao prompt final
                prompt_final += f"---\nA pergunta do usuário é: {prompt}\n"

                #st.markdown('Resposta baseada no arquivo')
                #st.markdown(f'Resposta usando RetrievalQA: {qa.run(prompt)}')
            else:
                st.warning("A chave do Pinecone não foi informada, a conversa não utilizará o arquivo pdf como base de conhecimento.")
                prompt_final = f"""
                Contexto: {st.session_state.mensagens}
                Responda com um estilo {estilo}: {estilo_selecionado}
                Responda com no máximo {tamanho} tokens sem perder o sentido da resposta.
                Pergunta: {prompt}

                """
        else:
            if prompt:    
                #st.markdown('Resposta baseada no chat')
                prompt_final = f"""
                Contexto: {st.session_state.mensagens}
                Responda com um estilo {estilo}: {estilo_selecionado}
                Responda com no máximo {tamanho} tokens sem perder o sentido da resposta.
                Pergunta: {prompt}

                """
        #Gerar a resposta
        if prompt:
            #custom_moderation = CustomModeration()
            custom_moderation = CustomModeration(openai_api_key=openai_api_key)
            moderation = custom_moderation.run(prompt)

            if moderation == '':
                #saida = llm(prompt_final)
                #with get_openai_callback() as cb:
                with get_openai_callback() as cb:    
                    saida = llm.invoke(prompt_final)
                    #st.markdown(cb)
                    tokens_utilizados = tokens_utilizados + cb.total_tokens
                    custo_estimado = custo_estimado + cb.total_cost
                st.session_state.mensagens.append({"role": 'user', "content": prompt})
                st.session_state.mensagens.append({"role": 'system', "content": saida})
                # Atualizando o histórico na interface
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("system"):
                    st.markdown(saida)
                

            else:
                #st.error(f'Moderação: A mensagem "{prompt}" não é apropriada. Tente novamente com outra mensagem.')
                st.error(moderation)
                
             
            c0,_,c1 = st.columns(3)
            with c1:
                st.button('Finalizar Conversa',
                    type = 'primary',
                    use_container_width = True,
                    on_click = lambda: finalizar(tokens_utilizados, custo_estimado, c0, index_name)) 
    else:
        if prompt:
            st.warning("Por favor, informe a chave da API OpenAI.")

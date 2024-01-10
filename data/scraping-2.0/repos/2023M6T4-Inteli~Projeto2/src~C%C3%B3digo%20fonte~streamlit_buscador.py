import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter
import emoji
import string
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

# Substituia com o token da API
llm = OpenAI(api_token="")

# create PandasAI object, passing the LLM
pandas_ai = PandasAI(llm)


# Função para renderizar a barra lateral com os links para as seções da sua aplicação
def render_sidebar():
    st.sidebar.title("CHAT-BTG")
    page = st.sidebar.radio("Navegue entre as seções", ["Upload do dataset", "Dashboard", "Chat-Btg", "Predição"])
    return page

# Função para ler o arquivo CSV 
def read_csv(upload_file):
    df = pd.read_csv(upload_file)
    
    return df
#baixar csv
def download_link(df, filename='predicoes.csv', link_text='Download CSV'):
    """Gera um hyperlink para baixar um DataFrame em formato CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


#Função para gerar gráfico de porcentagem de sentimento e histograma
#df = dataframe
#sentimento = coluna de sentimento
def percentage_hist_sentiment(df, sentimento):
    st.write('Gráfico de Pizza e Histograma')
    df = pd.DataFrame(df[sentimento].map({'POSITIVE':0, 'NEGATIVE':2, 'NEUTRAL':1}), columns=[sentimento])
    # Porcentagens
    # Positivo
    total = len(df)
    positivo = len(df.query(f'{sentimento} == 0')) / total
    # Neutro
    neutro = len(df.query(f'{sentimento} == 1')) / total
    # Negativo
    negativo = len(df.query(f'{sentimento} == 2')) / total
    # Criar DataFrame para o gráfico de barras
    # Plotar o gráfico de barras
    st.subheader(f'Quantidade Total:  {len(df)}')
    fig = px.pie(df, values=[positivo, neutro, negativo], names=['Positivo', 'Neutro', 'Negativo'])
    st.plotly_chart(fig)
    #Histograma
    labels = df[sentimento].map({0 :'Positivo', 1: 'Neutro', 2: 'Negativo'})
    fig_hist = px.histogram(df, labels)
    st.plotly_chart(fig_hist)

#Gráfico word cloud positivo
def word_cloud_positive(df, texto, sentimento):
    st.write("Gráfico Word Cloud Positiva")
    df_1 = pd.DataFrame(df[sentimento].map({'POSITIVE':0, 'NEGATIVE':2, 'NEUTRAL':1}), columns=[sentimento])
    df_1[texto] = df[texto].astype(str)

    # Pegando apenas as frases com o sentimento correspondente
    filtered_df = df_1.query(f'{sentimento} == 0')
    # Transformando em tokens
    tokens = filtered_df[texto].apply(lambda x: str(x).split())
    # Array para armazenar
    words = []
    # Looping para salvar os tokens em um array
    for sentences in tokens:
        for token in sentences:
            words.append(token)
    # Concatenar tudo em uma string
    text = ' '.join(words)

    # Criar um objeto WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Plotar a nuvem de palavras com Plotly Express
    fig = px.imshow(wordcloud.to_array())
    fig.update_layout(
        showlegend=False, width=800, height=400, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    st.plotly_chart(fig)
    
#Gráfico word cloud negativo
def word_cloud_negative(df,texto, sentimento):
    st.write("Gráfico Word Cloud Negativa")
    # Pegando apenas as frases com o sentimento correspondente
    df_3 = pd.DataFrame(df[sentimento].map({'POSITIVE':0, 'NEGATIVE':2, 'NEUTRAL':1}), columns=[sentimento])
    df_3[texto] = df[texto].astype(str)

    filtered_df = df_3.query(f'{sentimento} == 2')
    # Transformando em tokens
    tokens = filtered_df[texto].apply(lambda x: str(x).split())
    # Array para armazenar
    words = []
    # Looping para salvar os tokens em um array
    for sentences in tokens:
        for token in sentences:
            words.append(token)
    # Concatenar tudo em uma string
    text = ' '.join(words)

    # Criar um objeto WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Plotar a nuvem de palavras com Plotly Express
    fig = px.imshow(wordcloud.to_array())
    fig.update_layout(
        showlegend=False, width=800, height=400, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    st.plotly_chart(fig)


def count_emojis(df, text):
    texto = "".join(df[text]).lower()
    count = emoji.emoji_count(texto) # Contar a quantidade de emojis 
    emoji_dict = dict(Counter(c for c in texto if emoji.is_emoji(c))) # Contagem de emojis do dicionario

    most_common_emojis = Counter(emoji_dict).most_common(15) # Top 15 emojis mais utilizados nos comentários

    total_emojis = sum(emoji_dict.values()) # Cálculo da porcentagem de aparição de cada emoji
    emoji_percentages = {k: v / total_emojis for k, v in most_common_emojis}

    df = pd.DataFrame({'emoji': list(emoji_percentages.keys()), 'percentage': list(emoji_percentages.values())}) # Dataframe dos resultados

    df = df.sort_values(by='percentage', ascending=False)# Ordena os resultados por porcentagem decrescente
    
    # Criar o gráfico de barras
    fig = px.bar(df, x='emoji', y='percentage', labels={'emoji': 'Emoji', 'percentage': 'Porcentagem'})

    # Exibir o gráfico
    st.plotly_chart(fig)

# Extração de palavras que mais aparecem (gráfico de frequência)
def scatter_plot():
    # Transformando em tokens
    tokens = df['texto_pre'].astype(str).apply(lambda x: x.split())
    # Array para armazenar
    words = []
    # Looping para salvar os tokens em um array
    for sentences in tokens:
        for token in sentences:
            words.append(token)
    # Concatenar em uma string
    text = ' '.join(words)

    word_counts = Counter(text.split())
    top_words = word_counts.most_common(10)
    x = range(len(top_words))
    y = [count for word, count in top_words]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('Palavras')
    ax.set_ylabel('Frequência')
    ax.set_title('Gráfico de Dispersão de Palavras')

    # Adicionar rótulos das palavras
    for i, (word, count) in enumerate(top_words):
        ax.annotate(word, (x[i], y[i]))

    st.pyplot(fig)
    

# Barra de pesquisa
def search_bar():
    text_search = st.text_input("Pesquise palavras relacionadas à campanha", value="")
    df_search = df[df["texto_pre"].str.contains(text_search) | df["sentimento"].str.contains(text_search)]

    # Criação da tabela de palavras
    if not df_search.empty:
        words = []
        for _, row in df_search.iterrows():
            tokens = row['texto_pre'].split()
            words.extend(tokens)
        word_counts = pd.Series(words).value_counts().reset_index()
        word_counts.columns = ['Palavra', 'Frequência']

        # Ordenação das palavras por frequência
        word_counts = word_counts.sort_values(by='Frequência', ascending=False)


        st.dataframe(word_counts.style.set_properties(**{'overflow': 'scroll', 'width': '100%', 'max-height': '400px'}))
    else:
        st.write("Nenhum resultado encontrado.")

#Load no modelo
# model = torch.load('C:\\Users\\Inteli\\Downloads\\modelo_BERT\\model.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("C:\\Users\\Inteli\\Downloads\\model.pt", map_location=torch.device('cpu'))

#função para realizar predição da frase
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
def predict(model, tokenizer, sentence):
    model.eval()

    inputs = tokenizer.encode_plus(
        sentence,
        None,
        add_special_tokens=True,
        max_length=200,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    ids = ids.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        outputs = model(ids, token_type_ids=None, attention_mask=mask)

    outputs = outputs[0].detach().cpu().numpy()
    predict_code = np.argmax(outputs, axis=1)[0]

    inversed_map = {0: 'POSITIVE', 1: 'NEUTRAL', 2: 'NEGATIVE'}
    return inversed_map[predict_code]

#retirar stop_words
def stop(texto):
    if isinstance(texto, str):
        stop_words = ['eu', 'tu', 'ele', 'ela', 'nós', 'nos', 'vós', 'vos', 'eles', 'elas', 'um', 'em', 'de'
                    'isso', 'isto', 'aquilo', 'algum', 'alguma', 'alguns', 'algumas',
                    'outro', 'outra', 'outros', 'outras', 'tão', 'tal', 'tanto', 'tanta', 'tantos', 'tantas', 
                    'seu', 'sua', 'seus', 'suas', 'dele', 'dela', 'deles', 'delas', 
                    'quem', 'qual', 'quais', 'que', 'onde', 'como', 'e','um', 'as', 'no',
                    'para', 'por', 'com', 'sem', 'sob', 'sobre', 'de', 'da', 'desde', 'em', 'entre', 'porque',
                    'á', 'a', 'o', 'ola', 'olá', 'pra', 'para', 'bemvindo', 'benvindo', 'bem-vindo', 'bemvindos', 'aqui', 'vai', 'na', 'no', 'esse', 'este', 'voce', 'nosso', 'ou', 'btg','ser', 'mais', 'ter', 'meu', 'se', 'esta', 'todo', 'estar']
        

    new = []
    for word in texto.split():
        if word not in stop_words:
            new.append(word)
    return ' '.join(new)

#função para realizar predição da coluna
def add_prediction_column(df,text, model, tokenizer):
    df.dropna(inplace = True)
    
    df['Predicao'] = df[text].apply(lambda x: predict(model, tokenizer, x))
    return df


# Renderiza a barra lateral
page = render_sidebar()

if page == "Upload do dataset":
    #Título Página
    st.title("Upload dos dados")
    st.subheader('Visualização dos dados presentes na base inserida.')
    # Variável para armazenar o arquivo CSV
    upload_file = st.file_uploader('Escolha o seu arquivo CSV', type='csv')

    # Condição para colocar na tela o arquivo CSV
    if upload_file:
        st.markdown('---')
        # Lê os dados do arquivo CSV e armazena em um cache persistente
        df = read_csv(upload_file)
        st.dataframe(df)
        # success
        st.success("Success")        
# Mostra a página correspondente à opção selecionada na barra lateral
elif page == "Dashboard":
    #Título Página
    st.title("DASHBOARD")
    st.subheader('Identificação de porcentagens de sentimento, Word Cloud, frequência de emojis ou frequência de palavras na base de dados.')
    # Variável para armazenar o arquivo CSV
    upload_file = st.file_uploader('Escolha o seu arquivo CSV', type='csv')
    # Condição para colocar na tela o arquivo CSV
    if upload_file:
        st.markdown('---')
        # Lê os dados do arquivo CSV e armazena em um cache persistente
        df = read_csv(upload_file)

        if search_bar():
            st.write('yep')

        # Select box para os gráficos que gostaria de analisar
        plot_select = st.selectbox(
            'O que você gostaria de analisar?',
            ('Porcentagem de Sentimento', 'Word Cloud','Emojis que mais aparecem', 'Frequência das palavras')
        )

        # Mostrar gráfico correspondente à opção selecionada
        if plot_select == 'Porcentagem de Sentimento':
            percentage_hist_sentiment(df, 'sentimento')

        elif plot_select == 'Word Cloud':
            #word cloud positiva
            word_cloud_positive(df, 'texto_pre', 'sentimento')
            #word cloud negative
            word_cloud_negative(df, 'texto_pre', 'sentimento')


        elif plot_select == 'Emojis que mais aparecem':
            # Código para criar o gráfico de dispersão
            st.write("Gráfico de barra")

            count_emojis(df, 'texto')

        elif plot_select == 'Frequência das palavras':
            # Código para criar o gráfico de dispersão
            st.write("dispersão")
            scatter_plot()


elif page == "Chat-Btg":
    # Título da página
    st.title('CHAT-BTG')
    st.subheader('Teste o modelo BERT de predição de sentimentos.')
    
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Coloque seu texto aqui")
        submit_button = st.form_submit_button(label='Fazer análise')

    #Quando executar o botão, realiza o pre-processamento
    if submit_button:
        nlp = spacy.load("pt_core_news_lg")

        # Remover pontuação do texto inserido pelo usuário
        raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))
        #substitui emojis por palavras
        raw_text = emoji.demojize(raw_text, delimiters=(' ', ' '), language='pt')
        #retirando stop_words
        raw_text = stop(raw_text)
        # Lematizar o texto
        doc = nlp(raw_text)
        lemmas = [token.lemma_ for token in doc]
        preprocessed_text = " ".join(lemmas)
        
        # Fazer a predição na entrada lematizada
        prediction = predict(model, tokenizer, preprocessed_text)
        st.write('O sentimento do texto é', prediction)
        st.write('Texto pré-Processado:', preprocessed_text)
        st.write('Tokens:', preprocessed_text.split())
    st.title("Análise de dados por texto com Pandas AI")
    uploaded_file = st.file_uploader("Insira o arquivo CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(3))
        prompt = st.text_area("Digite a análise que deseja")

        # Generate output
        if st.button("Gerar análise"):
            if prompt:
                # call pandas_ai.run(), passing dataframe and prompt
                with st.spinner("Gerando a resposta..."):
                    st.write(pandas_ai.run(df, prompt))
            else:
                st.warning("Por favor, coloque uma análise.")


elif page == "Predição":
    st.title("PREDIÇÃO")
    st.subheader("Realize a predição de sentimentos da sua base de dados.")
    #Instanciando arquivo para importação
    upload_file_2 = st.file_uploader('Escolha o seu arquivo CSV', type='csv')
    #botão para executar a função
    if upload_file_2:
        df_2 = pd.read_csv(upload_file_2)
        st.dataframe(df_2)
        
        pred_button = st.button(label='Predição')
        if pred_button:
            df_2 = add_prediction_column(df_2, "texto", model, tokenizer)
            st.dataframe(df_2)
            st.markdown(download_link(df_2), unsafe_allow_html=True)

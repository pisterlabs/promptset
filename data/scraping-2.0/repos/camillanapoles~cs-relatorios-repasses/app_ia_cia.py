import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Sales Analysis App",
    page_icon=":sales:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define styles for the app
styles = """
<style>
img {
    max-width: 50%;
}
.sidebar .sidebar-content {
    background-color: #f5f5f5;
}
</style>
"""

import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="App - Analise de Repassess",
    page_icon=":sales:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define styles for the app
styles = """
<style>
img {
    max-width: 50%;
}
.sidebar .sidebar-content {
    background-color: #f5f5f5;
}
</style>
"""

# Render styles
st.markdown(styles, unsafe_allow_html=True)

#image = Image.open(r"C:\Downloads\20-Easy-Call-Center-Sales-Tips-to-Increase-Sales-1024x536.png")
#image2 = Image.open(r"C:\Downloads\sales-prediction.jpg")

# Define header
header = st.container()
with header:
    #st.image(image)
    st.title("Cia do Sorriso - Analise de Repasses")
    st.markdown("presente de Camilla Naooles para Antonio Sa")
    st.write("")
    

# Define main content
content = st.container()
with content:
    # Load sales dataset
    sale_file = st.file_uploader('Selecione seu CSV (fornecido por padrão)')
    if sale_file is not None:
        df = pd.read_csv(sale_file, encoding='latin-1')
    else:
        st.warning("Selecione um arquivo CSV para continuar.")
        st.stop()

    # Select x and y variables
    st.subheader("Crie seu gráfico")
    st.write("Selecione as variaveis x e y para criar um gráfico de dispersão.")
    col1, col2 = st.beta_columns(2)
    with col1:
        selected_x_var = st.selectbox('X variable', ['data_virada', 'ultimo_fir_updated_at', 'data_pagamento'])
    with col2:
        selected_y_var = st.selectbox('Y variable', ['repasse_liberado', 'valor_pago_ati', 'valor_glosado_fir'])

    # Create scatterplot
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x = df[selected_x_var], y = df[selected_y_var], hue = df['dentistas'])
    plt.xlabel(selected_x_var)
    plt.ylabel(selected_y_var)
    plt.title("Scatterplot of Sales")
    st.pyplot(fig)
    
# Define sidebar
sidebar = st.sidebar
with sidebar:
    st.image(image2)
    st.subheader("Obtenha insights sobre os dados")
    st.write("Insira uma pergunta para gerar insights sobre os dados usando inteligencia artificial")
    prompt = st.text_input("escreva aqui o que deseja:")
    if prompt:
        # Initialize PandasAI and OpenAI
        llm = OpenAI()
        pandas_ai = PandasAI(llm)
        
        # Run PandasAI with user input prompt
        result = pandas_ai.run(sale_df, prompt=prompt)
        
        # Display result
        if result is not None:
            st.write("### Insights")
            st.write(result)

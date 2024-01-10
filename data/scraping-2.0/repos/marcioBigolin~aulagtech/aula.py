import streamlit as st
import pandas as pd

from sqlalchemy import create_engine

st.set_page_config(page_title="MDI - Análise com IA", page_icon="🌍", layout="wide")



# Recebe os parâmetros via GET enquanto sem criptografia mandando direto (usar bearertok)
params = st.experimental_get_query_params()
# Obtém o valor do parâmetro 'variavel' da URL
senha = params.get('senha', ['SEM_DADOS'])[0]

#montando a tela 
st.write("Escolha uma das abas")

def gepeto():
    from pandasai import SmartDataFrame
    from pandasai.llm.openai import OpenAI
    import matplotlib.pyplot as plt
    import os

    api_token = "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"                
    st.session_state.prompt_history = []


    
    with st.form("Question"):
        question = st.text_input("Question", value="", type="default")
        submitted = st.form_submit_button("Gerar")
        if submitted:
            with st.spinner():
                llm = OpenAI(api_token=api_token)
                pandas_ai = PandasAI(llm)
                x = pandas_ai.run(df, prompt=question)

                if os.path.isfile('temp_chart.png'):
                    im = plt.imread('temp_chart.png')
                    st.image(im)
                    os.remove('temp_chart.png')

                if x is not None:
                    st.write(x)

                    st.session_state.prompt_history.append(question)
  

        st.subheader("Prompt history:")
        st.write(st.session_state.prompt_history)

        if "prompt_history" in st.session_state.prompt_history and len(st.session_state.prompt_history) > 0:
            if st.button("Limpar"):
                st.session_state.prompt_history = []
                st.session_state.df = None


if senha == 'SEM_DADOS':
    st.write("Sem dados")
else:
    #criando conexão com o banco de dados
    conn = create_engine(f"postgresql://revisao_data:{senha}@revisao_data.postgresql.dbaas.com.br:5432/revisao_data")

    #consultando o banco de dados (conhecimento de SQL)
    sql_query =  pd.read_sql_query (f"SELECT * FROM moodle_marcio2.fato_join;", con=conn)

    df = pd.DataFrame(sql_query, columns = ['titulo', 'nome_completo', 'coh_frazier', 'coh_brunet', 'data_entrega'])

    tab1, tab2, tab3 = st.tabs(["Entendendo meus dados", "Gerador de gráfico", "ChatGPT" ])

    with tab1:
        st.dataframe(df)
        st.write(df['coh_brunet'].describe())


    with tab2: 
        import pygwalker as pyg
        from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
        # Establish communication between pygwalker and streamlit
        init_streamlit_comm()

        @st.cache_resource
        def get_pyg_renderer() -> "StreamlitRenderer":
            # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
            return StreamlitRenderer(df, spec="./gw_config.json", debug=False)
    
        renderer = get_pyg_renderer()
    
        # Render your data exploration interface. Developers can use it to build charts by drag and drop.
        renderer.render_explore()
    with tab3:
        gepeto()
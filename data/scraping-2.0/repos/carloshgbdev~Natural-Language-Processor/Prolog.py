import openai
import streamlit as st

st.title("Natural Language Processor")
st.write("Bem vindo ao nosso projeto um transformador de linguagem natural em Prolog. Para começar, digite uma frase em português e clique para ver a tradução em Prolog.")
openai.api_key = "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"

# Defino que na interface terá um local onde o usuário poderá inputar o que ele gostaria de perguntar
input = st.text_area("Frase em Linguagem Natural: ",label_visibility = "visible",height = 4)

# Remova a vírgula após prompt_default para que seja uma única string
prompt_default = f"Translate Natural Language Sentences into clauses to be used in the Prolog language \n Text {input} \n Translation"

# Primeiro iremos criar uma função para que possamos mostrar para nosso modelo o que ele deve ser treinado para realizar
def Modelo_transformando(prompt_inp):
    resposta = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_inp,
        temperature=0.3,
        max_tokens=200,
        top_p=1,
        presence_penalty = 0.0
    )
    return resposta["choices"][0]["text"].strip()

if st.button('Submit'):
    prompt_inp = prompt_default.format(input)
    output = Modelo_transformando(prompt_inp)
    st.write("Frase em Linguagem Prolog: ")  
    st.write(output)
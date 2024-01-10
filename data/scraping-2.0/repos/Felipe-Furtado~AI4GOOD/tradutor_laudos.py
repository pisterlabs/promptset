import streamlit as st
import easyocr
from PIL import Image
import tempfile
import openai
import os

# Initialize EasyOCR with desired languages
@st.cache_resource
def load_model():
    reader = easyocr.Reader(['pt', 'en'], gpu=False)
    return reader

reader = load_model()

st.header("Elucidativa - Explicadora de Laudos de Exames Médicos")
st.markdown(
    "##### Esse aplicativo ajuda você a entender os resultados de exames médicos. Envie uma foto do seu laudo e receba uma explicação em linguagem simples e acessível."
    )

# File uploader for image selection
laudo_original = st.file_uploader("Selecione a foto do laudo que deseja esclarecer", type=['png', 'jpg', 'jpeg'])

# Define function to process image on button click
@st.cache_data(show_spinner="Extraindo texto do laudo..." )
def process_image():
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(laudo_original.read())

        # Close the temporary file to release the file handle
        temp_image.close()

        # Read the image using PIL
        image = Image.open(temp_image.name)

        # Perform OCR on the image using EasyOCR
        # Extract text from the temporary image file
        with st.spinner("Extraindo texto do laudo..."):
            text_results = reader.readtext(temp_image.name, detail=0)

        # Combine the list of strings into a paragraph
        # Join the list elements with a space
        texto_laudo = ' '.join(text_results)
    return texto_laudo

if laudo_original is not None:
    if st.button("Enviar"):
    # Display the OCR result as a paragraph
        texto_laudo = process_image()
        expander = st.expander("Texto Extraído do Laudo Original")
        expander.write(texto_laudo)

# LLM integration

#TODO Comment & remove API Key after local testing
#os.environ["OPENAI_API_KEY"] = "sk-...
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI()
if "texto_laudo" in locals():
        with st.spinner("Traduzindo laudo..."):
            llm_call = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                    "role": "system",
                    "content": 
                    """
                    Você é um sistema especializado em explicar termos médicos e científicos de forma acessível a pessoas leigas. Sua missão é receber um laudo de exame médico e fornecer uma explicação concisa sobre os achados descritos para o paciente, utilizando uma linguagem simples e compreensível para um estudante do ensino fundamental.\n\nEstrutura do texto de saída:\n\nResumo dos resultados:Inicie com um parágrafo resumindo o propósito do exame e destacando se há algum achado anormal. Se não houver, informe que nada de errado foi identificado.\n\nExplicação da gravidade:Caso haja algum achado anormal, explique a gravidade de maneira clara e simples, enfatizando que a palavra final sempre deve vir do médico responsável pelo caso.\n\nAchados críticos (se aplicável):Se houver achados críticos que necessitem atenção imediata, chame atenção com emojis e incentive a pessoa a marcar uma consulta de retorno o mais rápido possível.
                    """
                    },
                    {
                    "role": "user",
                    "content": texto_laudo
                    }
                ],
                temperature=.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
        if llm_call is not None:
            st.success("Tradução concluída!")
            st.cache_data.clear()
            resposta = llm_call.choices[0].message.content
            st.write(resposta)
             
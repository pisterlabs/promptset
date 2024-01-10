import streamlit as st
from PIL import Image
from whisperx_detect import detect_lyrics
from generate_sd_prompts import generate_sd_prompts
from generate_video import generate_video
from langchain_FAISS import extract_prompt_FAISS
from generate_streamlit_imgs import generate_streamlit_imgs
import openai
import os
import tempfile

os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.environ['OPENAI_API_KEY']

# DEFINIMOS LA CONEXIÓN CON CHATGPT
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]

def main():

    st.markdown("""
    <style>
    body {
        background-image: url("https://weirdwonderfulai.art/wp-content/uploads/2022/08/stablediffusion-study.png");
        background-size: cover;
        opacity: 0.85;
    }
    </style>
    """, unsafe_allow_html=True)
    
    logo = Image.open('./utils/logo.png')
    st.image(logo)
    st.header('Bienvenido a VerseVisions, una aplicación que permite visualizar el contenido de una canción en forma de video utilizando Stable Diffusion.')

    st.subheader('Para comenzar, seleccione una canción y un estilo artístico.')
    st.subheader('Los campos marcados con * son obligatorios.')

    uploaded_file = st.file_uploader("Seleccione una canción *", type=['mp3', 'wav', 'flac'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/ogg')
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.getvalue())
        tfile.close()

    options = ['realistic', 'futuristic', 'professional portrait', 'cyberpunk', 'animation', 
               'painting', 'pencil drawing', 'anime', 'vintage portrait', 'hyper realistic', 'matte painting']
    selected_option = st.selectbox('Seleccione estilo *', options)

    artistic_input = st.text_input("Introduzca alguna connotación artística (opcional)")

    output_dir = st.text_input("Introduzca el nombre de la carpeta donde se almacerán los archivos generados *")

    def process(tfile, selected_option, artistic_input, output_dir):
        # Agrega aquí el código para procesar el archivo de audio
        lyrics = detect_lyrics(tfile.name)

        song_prompt = f"""Try to identify the song that the lyrics below are from. \
        maybe all lyrics not match at all, but try to find the closest one. \
        if you have the song, your answer should be exclusively the title of the song and the artist. \
        the lyrics are delimited by triple backticks:
        ```{lyrics}```
        """
        song_title = get_completion(song_prompt)
        st.write('La canción introducida es: ', song_title)

        lyrics_prompt =f"""give me the full lyrics of {song_title}\
        just provide the full lyrics, not anything else, like the title of the song or the artist,
        and no extra comments.
        """
        response_lyrics = get_completion(lyrics_prompt)
        final_lyrics = [x for x in response_lyrics.split("\n") if x]
        image_prompt_faiss = extract_prompt_FAISS(style=str(selected_option + ' ' + artistic_input))
        image_prompt_faiss = image_prompt_faiss[8:]
        st.write('Se ha escogido el siguiente prompt como plantilla: ', image_prompt_faiss)

        prompts_list = generate_sd_prompts(final_lyrics, image_prompt_faiss)

        st.write('Se han generado los siguientes prompts: ', prompts_list)

        generate_streamlit_imgs(prompts_list, output_dir)

        archivos = os.listdir(output_dir)
        imagenes = [os.path.join(output_dir, imagen) for imagen in archivos]

        output_dir_abs_path = os.path.abspath(output_dir)
        output_file_abs_path = os.path.join(output_dir_abs_path, f"{output_dir}.mp4")

        generate_video(imagenes, output_file_abs_path, fps=60, espera=2)

        st.write('Se ha generado el siguiente video: ', output_file_abs_path)

        return output_file_abs_path

    if uploaded_file is not None and selected_option and output_dir:
        if st.button("Iniciar procesamiento"):
            video_path = process(tfile, selected_option, artistic_input, output_dir)
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.write("El procesamiento se ha completado y el video se ha generado con éxito.")
    else:
        st.write("Por favor, rellene todos los campos requeridos (canción, estilo y directorio de salida) para iniciar el procesamiento.")

if __name__ == "__main__":
    main()

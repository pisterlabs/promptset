import streamlit as st
from PIL import Image
from whisperx_detect import detect_lyrics
from generate_video import generate_video
from generate_streamlit_imgs import generate_streamlit_imgs
from get_lyrics import get_lyrics
from add_music_text import add_music_text
from pydub.utils import mediainfo
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
        opacity: 0.89;
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

    options = ['Animal','Archviz','Building','Cartoon Character','Concept Art / Design',
               'Cyberpunk','Digital Art','Digital Art Landscape','Drawing','Fashion',
               'Landscape','Photograph Closeup', 'Photograph Portrait', 'Postapocalyptic',
               'Schematic','Sketch','Space','Sprite','Steampunk','Vehicles']
    
    prompt_templates = {'Animal':'{i}, wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols',
                        'Archviz': '{i}, by James McDonald and Joarc Architects, home, interior, octane render, deviantart, cinematic, key art, hyperrealism, sun light, sunrays, canon eos c 300, ƒ 1.8, 35 mm, 8k, medium - format print',
                        'Building':'{i}, shot 35 mm, realism, octane render, 8k, trending on artstation, 35 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, realistic matte painting, hyper photorealistic, trending on artstation, ultra - detailed, realistic',
                        'Cartoon Character':"{i},anthro, very cute kid's film character, disney pixar zootopia character concept artwork, 3d concept, detailed fur, high detail iconic character for upcoming film, trending on artstation, character design, 3d artistic render, highly detailed, octane, blender, cartoon, shadows, lighting",
                        'Concept Art / Design':"Design {i}, character sheet, concept design, contrast, style by kim jung gi, zabrocki, karlkka, jayison devadas, trending on artstation, 8k, ultra wide angle, pincushion lens effect",
                        'Cyberpunk':"{i}, cyberpunk, in heavy raining futuristic tokyo rooftop cyberpunk night, sci-fi, fantasy, intricate, very very beautiful, elegant, neon light, highly detailed, digital painting, artstation, concept art, soft light, hdri, smooth, sharp focus, illustration, art by tian zi and craig mullins and wlop and alphonse much",
                        'Digital Art':'{i}, ultra realistic, concept art, intricate details, highly detailed, photorealistic, octane render, 8k, unreal engine, sharp focus, volumetric lighting unreal engine. art by artgerm and alphonse mucha',
                        'Digital Art Landscape':'{i}, epic concept art by barlowe wayne, ruan jia, light effect, volumetric light, 3d, ultra clear detailed, octane render, 8k, dark green, light green colour scheme',
                        'Drawing': '{i}, cute, funny, centered, award winning watercolor pen illustration, detailed, disney, isometric illustration, drawing, by Stephen Hillenburg, Matt Groening, Albert Uderzo',
                        'Fashion':'photograph of a Fashion model, {i}, full body, highly detailed and intricate, golden ratio, vibrant colors, hyper maximalist, futuristic, city background, luxury, elite, cinematic, fashion, depth of field, colorful, glow, trending on artstation, ultra high detail, ultra realistic, cinematic lighting, focused, 8k',
                        'Landscape':'{i}, birds in the sky, waterfall close shot 35 mm, realism, octane render, 8 k, exploration, cinematic, trending on artstation, 35 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, moody cinematic epic concept art, realistic matte painting, hyper photorealistic, epic, trending on artstation, movie concept art, cinematic composition, ultra - detailed, realistic',
                        'Photograph Closeup':'{i}, depth of field. bokeh. soft light. by Yasmin Albatoul, Harry Fayt. centered. extremely detailed. Nikon D850, (35mm|50mm|85mm). award winning photography.',
                        'Photograph Portrait':'portrait photo of {i}, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography',
                        'Postapocalyptic':'{i}, fog, animals, birds, deer, bunny, postapocalyptic, overgrown with plant life and ivy, artgerm, yoshitaka amano, gothic interior, 8k, octane render, unreal engine',
                        'Schematic':'23rd century scientific schematics for {i}, blueprint, hyperdetailed vector technical documents, callouts, legend, patent registry',
                        'Sketch':'{i}, sketch, drawing, detailed, pencil, black and white by Adonna Khare, Paul Cadden, Pierre-Yves Riveau',
                        'Space':'{i}, by Andrew McCarthy, Navaneeth Unnikrishnan, Manuel Dietrich, photo realistic, 8 k, cinematic lighting, hd, atmospheric, hyperdetailed, trending on artstation, deviantart, photography, glow effect',
                        'Sprite':'sprite of video games {i} icons, 2d icons, rpg skills icons, world of warcraft, league of legends, ability icon, fantasy, potions, spells, objects, flowers, gems, swords, axe, hammer, fire, ice, arcane, shiny object, graphic design, high contrast, artstation',
                        'Steampunk':'{i}, steampunk cybernetic biomechanical, 3d model, very coherent symmetrical artwork, unreal engine realistic render, 8k, micro detail, intricate, elegant, highly detailed, centered, digital painting, artstation, smooth, sharp focus, illustration, artgerm, Caio Fantini, wlop',
                        'Vehicles':'photograph of {i}, photorealistic, vivid, sharp focus, reflection, refraction, sunrays, very detailed, intricate, intense cinematic composition'}

    generacion_automatica = st.checkbox('Estilo de generación de imágenes automático en función de la letra de la canción. (Nota: no todas las imágenes tendrán el mismo estilo, dependerá de la connotación de cada frase de la canción.)')

    if not generacion_automatica:
        selected_option = st.selectbox('Seleccione estilo *', options)
    else:
        selected_option = None

    output_dir = st.text_input("Introduzca el nombre de la carpeta donde se almacerán los archivos generados. El nombre introducido no puede tener esapacios en blanco. *")

    # comprobamos que el nombre de la carpeta no tenga espacios en blanco
    if output_dir:
        if " " in output_dir:
            st.write("El nombre de la carpeta no puede tener espacios en blanco, por favor, introduzca otro nombre válido.")
            return

    def process(tfile, selected_option, output_dir):

        detected_results = detect_lyrics(tfile.name)

        lyrics = [dic["text"] for dic in detected_results["segments"]]

        song_prompt = f"""Try to identify the song that the lyrics below are from. \
        maybe all lyrics not match at all, but try to find the closest one. \
        if you have the song, your answer should be exclusively the title of the song and the artist. \
        the lyrics are delimited by triple backticks:
        ```{lyrics}```
        """
        song_title = get_completion(song_prompt)
        st.write('La canción introducida es: ', song_title)

        titulo_cancion = song_title.split('by')[0]
        artista = song_title.split('by')[1]

        final_lyrics = get_lyrics(artista, titulo_cancion)

        #st.write('Las letras de la canción son: ', final_lyrics)

        start_time = detected_results['segments'][0]['start']

        info = mediainfo(tfile.name)
        duration = info['duration']

        start_frame = int(start_time * 12) # CAMBIAR FPS AQUÍ

        resto_tiempo = float(duration) - start_time

        frames_resto = resto_tiempo * 12 # CAMBIAR FPS AQUÍ

        frames_cada_frase = frames_resto / len(final_lyrics)

        frames_list = [0]

        cont_1 = 1

        for i in final_lyrics[1:]:
            frames_list.append(int(start_frame)+frames_cada_frase*cont_1)
            cont_1 += 1

        prompts_list = []

        cont = 0
        for i in final_lyrics:
            prompts_list.append(prompt_templates[selected_option].replace('{i}', i))
            cont += 1

            # if cont == 3:
            #     break

        generate_streamlit_imgs(prompts_list, output_dir)

        archivos = os.listdir(output_dir)
        imagenes = [os.path.join(output_dir, imagen) for imagen in archivos]

        output_dir_abs_path = os.path.abspath(output_dir)
        output_file_abs_path = os.path.join(output_dir_abs_path, f"{output_dir}.mp4")

        generate_video(imagenes, output_file_abs_path, fps=30, espera=2)

        add_music_text(video_path = output_file_abs_path, audio_path = tfile.name,lyrics=final_lyrics, lyric_frames = frames_list, out_dir= output_dir)

        os.system(f"ffmpeg -i {output_dir}/final_video.mp4 -vcodec libx264 {output_dir}/converted_final_video.mp4")

        return str(output_dir + "/converted_final_video.mp4")

    if uploaded_file is not None and output_dir:
        if not generacion_automatica and selected_option or generacion_automatica:
            if st.button("Iniciar procesamiento"):
                video_path = process(tfile, selected_option, output_dir)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                st.write("El procesamiento se ha completado y el video se ha generado con éxito.")
        else:
            st.write("Por favor, seleccione un estilo para iniciar el procesamiento.")
    else:
        st.write("Por favor, rellene todos los campos requeridos (canción, estilo y directorio de salida) para iniciar el procesamiento.")

if __name__ == "__main__":
    main()

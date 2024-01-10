import os
import openai
import tempfile
from pydub import AudioSegment
from utils.solapamiento import Solapamiento
from ia.chatgpt import Chatgpt
from utils.audio import Audio
import uuid
import streamlit as st

class Whisper():

    def __init__(self, apis_key):
        self.apis_key = apis_key

        self.diez_minutos = 10 * 60 * 1000
        self.diez_seg = 10 * 1000
        self.unmin = 60 * 1000

    def transcribe(self, audio_file, language, translate):

        if language == "Ingles":
            lan = "en"
        else:
            lan = "es"

        file_path = self.crear_tempfile(audio_file)

        if Audio.valid_size(file_path):
            audio = AudioSegment.from_file(file_path)
            return self.llamada_a_whisper(file_path, audio, lan, translate)
        else:
            return self.transcribe_long(file_path, lan, translate)

    def transcribe_long(self,file_path, lan, translate):

        audio = AudioSegment.from_file(file_path)

        # Crea un segmento de audio silencioso de la misma duración que los fragmentos de audio
        silencio = AudioSegment.silent(duration=self.diez_seg)

        responses = []
        solapamientos = []
        salidas = []

        # TRANSCRIBE EL AUDIO EN FRAGMENTOS DE 10 EN 10 MINUTOS
        for i, audio_part in enumerate(audio[::self.diez_minutos]):

            fragmento_path = f"fragmento{i}.mp3"

            # Concatena el fragmento de audio con el segmento de audio silencioso
            audio_con_silencio = audio_part + silencio

            response = self.llamada_a_whisper(fragmento_path, audio_con_silencio, lan, translate)

            responses.append(response)

            #st.write(f"Fragmento {i}: {response}")

            # calcula el inicio y el fin del solapamiento entre 2 fragmentos sucesivos
            ini = (i+1)*self.diez_minutos - self.diez_seg
            end = (i+1)*self.diez_minutos + self.diez_seg

            if end < len(audio):

                #crea y exporta el fragmento de audio correspondiente al solapamiento
                solapamiento_audio = audio[ini:end]
                solapamiento_path = f"solapamiento"+ str(i) + "_" + str(uuid.uuid4().hex) + ".mp3"

                solap_response = self.llamada_a_whisper(solapamiento_path, solapamiento_audio, lan, translate)

                solapamientos.append(solap_response)

                #st.write(f"Solapamiento {i}: {solap_response}")

            #corta el fragmento anterior hasta la parte que marca el solapamiento
            if i > 0:
                prueba_izq, bool_izq = Solapamiento.recortar_texto(responses[i-1], solapamientos[i-1])
                prueba_drc, bool_drc = Solapamiento.recortar_texto(solapamientos[i-1], responses[i])

                if bool_drc & bool_izq:
                    prueba = prueba_izq + prueba_drc
                else:
                    prueba = responses[i-1]

                salidas.append(prueba)

                #st.write(prueba)

        #elimina fragmento de audio del servidor
        os.remove(file_path)
        salidas.append(response)

        # Junta todas las respuestas en una sola cadena de texto
        result = " ".join(salidas)

        return result

    def llamada_a_whisper(self, path, audio, lan, translate):

        audio.export(path, format="mp3")

        # abre el solapamiento de audio y se lo pasa a whisper
        with open(path, "rb") as audio_bytes:

            if not translate:
                response = openai.Audio.transcribe(
                    api_key=self.apis_key,
                    model="whisper-1",
                    file=audio_bytes,
                    response_format="text",
                    language = lan
                    # temperature=0.1
                )
            else:
                response = openai.Audio.translate(
                    api_key=self.apis_key,
                    model="whisper-1",
                    file=audio_bytes,
                    response_format="text"
                )

        os.remove(path)

        return response

    @staticmethod
    def crear_tempfile(audio_file):

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_file.read())
            file_path = temp_file.name
        return  file_path

    def result(self, contenido):

        # st.write(contenido)
        chat = Chatgpt(self.apis_key)
        respuesta = chat.respuesta(contenido)

        # Mostrar el contenido del archivo en Streamlit
        return respuesta

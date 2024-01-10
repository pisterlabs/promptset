"""
Este archivo contiene clases y funciones para cargar, procesar documentos de audio y transcribirlos.
Fue programado teniendo en mente su utilización en aplicaciones de IA.

Autor: Patrick Vásquez <pvasquezs@fen.uchile.cl>
Ultima actualización: 05/11/2023
"""

# Import modules

from abc import ABC, abstractmethod

# Format and name extracter
class ExtractDocumentFormat:
    def extract_extension(file_path):
        """
        Extrae la extensión del archivo. La api de openai soporta m4a, mp3, mp4, mpeg, mpga, wav y webm.
        :return: La extensión del archivo.
        """
        import os
        name, extension = os.path.splitext(file_path)
        return extension
    
class ExtractDocumentName:
    def extract_name(file_path):
        """
        Extrae la extensión del archivo. La api de openai soporta m4a, mp3, mp4, mpeg, mpga, wav y webm.
        :return: La extensión del archivo.
        """
        import os
        name, extension = os.path.splitext(file_path)
        return name
    
# Audio transcriber

class TranscribeAudio(ABC):

    @abstractmethod
    def transcribe_audio():
        pass

class OpenAISimpleTranscriber(TranscribeAudio):

    def __init__(self, api_key):
        self.api_key = api_key

    def transcribe_audio(self, file_path, model="whisper-1"):
        import openai
        file = open(file_path, "rb")
        transcription = openai.Audio.transcribe(model, file)
        return transcription
    
class OpenAISegmentsTranscriber(TranscribeAudio):
    def __init__(self, api_key):
        self.api_key = api_key

    def transcribe_audio(self, segment_paths):
        import openai
        """
        Transcribe una lista de segmentos de audio.
        :param segment_paths: Lista de rutas de archivos de segmentos de audio.
        :return: Lista de transcripciones correspondientes a los segmentos.
        """
        openai.api_key = self.api_key
        transcripciones = []
        # para cada segmento en la lista de rutas de segmentos
        for segment_path in segment_paths:
            # abrir el archivo de audio
            with open(segment_path, "rb") as audio_file:
                # transcribir el audio
                response = openai.Audio.create(
                    audio=audio_file,
                    content_type="audio/wav")
            # extraer la transcripcion del response
            transcripcion = response["transcriptions"][0]["text"]
            # agregar la transcripcion a la lista de transcripciones
            transcripciones.append(transcripcion)
        return transcripciones

# Audio splitter

class AudioSplitter:
    def split_audio(self, file_path, output_path = "output/audio", start_time = 0, segment_duration = 10000, overlap_duration = 1000):
        """
        Divide un archivo de audio en segmentos superpuestos.
        :param file_path: Ruta del archivo de audio.
        :param start_time: Tiempo de inicio en milisegundos (por defecto, 0ms).
        :param segment_duration: Duración de cada segmento en milisegundos (por defecto, 10000ms o 10 segundos).
        :param overlap_duration: Duración de la superposición en milisegundos (por defecto, 1000ms o 1 segundo).
        :return: Lista con los segmentos resultantes.
        """
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(file_path)
        end_time = segment_duration
        segment_paths = []

        # mientras el tiempo de termino sea menor o igual a la duracion del audio
        while end_time <= len(audio):
            # extraer el segmento del audio
            segmento = audio[start_time:end_time]
            # guardar la ruta y exportar el segmento en un archivo
            segment_path = f"{output_path}/segmento_{len(segment_paths)}.wav"
            segmento.export(segment_path, format="wav")
            # agregar la ruta a la lista de rutas
            segment_paths.append(segment_path)
            # actualizar el tiempo de inicio y termino
            start_time += segment_duration - overlap_duration
            end_time += segment_duration - overlap_duration
        # retornar la lista de rutas
        return segment_paths

# Audio editor

class AudioEditor:
    def noise_reduction(self, file_path, output_path = "processing/audio/processed_audio.wav"):
        """
        Reduce el ruido de un archivo de audio.
        :param file_path: Ruta del archivo de audio.
        :param output_path: Ruta del archivo de audio procesado.
        :return: Ruta del archivo de audio procesado.
        """
        import noisereduce as nr
        import soundfile as sf
        audio, sample_rate = sf.read(file_path)
        reduced_audio = nr.reduce_noise(y=audio, sr=sample_rate)
        sf.write(output_path, reduced_audio, sample_rate)
        return output_path
    
    def increase_volume(self, file_path, output_path = "processing/audio/processed_audio.wav"):
        """
        Aumenta el volumen de un archivo de audio.
        :param file_path: Ruta del archivo de audio.
        :param output_path: Ruta del archivo de audio procesado.
        :return: Ruta del archivo de audio procesado.
        """
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(file_path)
        audio = audio + audio
        audio.export(output_path, format="wav")
        return output_path

    def high_pass_filter(self, file_path, output_path="processing/audio/processed_audio.wav", cutoff_frequency=300):
        """
        Aplica un filtro pasa-altos a un archivo de audio para resaltar ciertas frecuencias.
        :param file_path: Ruta del archivo de audio.
        :param output_path: Ruta del archivo de audio procesado.
        :param cutoff_frequency: Frecuencia de corte para el filtro pasa-altos.
        :return: Ruta del archivo de audio procesado.
        """
        from scipy.signal import butter, lfilter
        import soundfile as sf
        audio, sample_rate = sf.read(file_path)

        def high_pass_filter(data, cutoff, sample_rate):
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff / nyquist
            b, a = butter(6, normal_cutoff, btype='high', analog=False)
            return lfilter(b, a, data)

        filtered_audio = high_pass_filter(audio, cutoff_frequency, sample_rate)
        sf.write(output_path, filtered_audio, sample_rate)
        return output_path
    
    def audio_normalization(self, file_path, output_path="processing/audio/processed_audio.wav"):
        """
        Normaliza el nivel de volumen de un archivo de audio.
        :param file_path: Ruta del archivo de audio.
        :param output_path: Ruta del archivo de audio procesado.
        :return: Ruta del archivo de audio procesado.
        """
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(file_path)
        normalized_audio = audio.normalize()
        normalized_audio.export(output_path, format="wav")
        return output_path
    

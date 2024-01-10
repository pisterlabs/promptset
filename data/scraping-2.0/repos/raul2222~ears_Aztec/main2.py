import os
import socket
import timeit
import tempfile
import wave
import threading
import queue
import collections
import requests
import json
import signal
import subprocess

import torch
from queue import Empty
from pydub import AudioSegment
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate 
from langchain.chat_models import ChatOpenAI
from faster_whisper import WhisperModel
from audio_utils import normalize_audio, compress_audio
from torch.hub import load


class AudioServer:
    def __init__(self, port, model_size='medium'):
        self.port = port
        self.model_size = model_size
        self.sampling_rate = 32000
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.model_id_whisper = 'whisper-1'
        self.model_id = 'gpt-4'
        self.modelo = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.model, self.utils = load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
        self.get_speech_timestamps, _, self.read_audio, *_ = self.utils
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = ('', port)
        self.frame_duration_ms = 100
        self.frame_length_samples = int(self.sampling_rate * self.frame_duration_ms / 1000)
        self.frame_length_bytes = self.frame_length_samples * 2
        self.buffer = b''
        self.write_data_lock = threading.Lock()
        self.output_file_count = 0
        self.data_queue = queue.Queue()
        self.post_speech_queue = collections.deque(maxlen=2)
        self.exit_event = threading.Event()

        # Asociar la función de manejo de señales al evento de interrupción (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Añadir el código para la cadena de lenguaje
        template = """Eres Aztec, un robot que interactua con niños autistas, 
        trabajas en una asociacion de niños autistas, proporcionas respuestas cortas,
        tienes mucho tacto con lo que dices porque estas rodeado de niños pequeños, 
        eres amable, agradable, simpatico, siempre estas dispuesto a ayudar a los demás,


        {chat_history}
        Human: {human_input}
        Chatbot:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], 
            template=template
        )
        memory = ConversationBufferMemory(memory_key="chat_history")

        self.llm_chain = LLMChain(
            llm=ChatOpenAI(temperature=0, model_name=self.model_id, openai_api_key=self.OPENAI_API_KEY), 
            prompt=prompt, 
            verbose=True, 
            memory=memory,
        )
    
    def bind(self):
        try:
            self.sock.bind(self.server_address)
        except OSError as e:
            if e.errno == 98:  # Address already in use
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    self.sock.bind(self.server_address)
                except OSError as e:
                    if e.errno == 98:  # Address still in use after killing processes
                        print(f"Port {self.port} is still in use. Killing processes using the port...")
                        self.kill_processes_using_port(self.port)
                        try:
                            self.sock.bind(self.server_address)
                        except OSError as e:
                            raise OSError(f"Failed to bind the socket to port {self.port}: {e}")
                    else:
                        raise OSError(f"Failed to bind the socket to port {self.port}: {e}")
            else:
                raise e

    def kill_processes_using_port(self, port):
        try:
            output = subprocess.check_output(["lsof", "-i", f":{port}"], universal_newlines=True)
            lines = output.strip().split("\n")[1:]
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    if pid.isdigit():
                        os.kill(int(pid), signal.SIGKILL)
        except subprocess.CalledProcessError:
            pass



    def start_server(self):
        self.bind()
        self.sock.listen(10)
        print("Server listening on {}:{}".format(*self.server_address))

        # Iniciar el hilo para escribir los datos en el archivo
        write_data_thread = threading.Thread(target=self.write_data_to_file)
        write_data_thread.start()

        # Loop principal del servidor
        try:
            while not self.exit_event.is_set():
                connection, client_address = self.sock.accept()
                print('Connection from', client_address)
                connection.settimeout(5.7)

                while not self.exit_event.is_set():
                    try:
                        data = connection.recv(4096)
                        if not data:
                            print("Client disconnected")
                            break

                        self.buffer += data

                        while len(self.buffer) >= self.frame_length_bytes:
                            frame_data = self.buffer[:self.frame_length_bytes]
                            self.buffer = self.buffer[self.frame_length_bytes:]
                            self.data_queue.put(frame_data)
                    except socket.timeout:
                        print("Connection timeout")
                        break

                connection.close()
        except KeyboardInterrupt:
            print("Server shutting down...")

        self.sock.close()
        self.exit_event.set()
        write_data_thread.join()

    def write_data_to_file(self):
        wav_file = None
        file_path = None
        non_speech_frames = collections.deque(maxlen=1)
    
        while True:
            with self.write_data_lock:
                if self.exit_event.is_set():
                    break

            try:
                frame_data = self.data_queue.get(timeout=1)
            except Empty:
                continue

            if wav_file is None:
                file_path = self.create_temporary_wav_file()
                wav_file = wave.open(file_path, "wb")
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sampling_rate)

            wav_file.writeframes(frame_data)

            print("write data to file")

            if len(non_speech_frames) == 1:
                for frame in non_speech_frames:
                    wav_file.writeframes(frame)
                wav_file.close()

                # Bloquear la escritura de datos mientras se procesa el audio
                with self.write_data_lock:
                    self.process_audio_file(file_path)

                wav_file = None
                non_speech_frames.clear()

    def create_temporary_wav_file(self):
        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
        temp_wav_file.close()
        return temp_wav_file.name

    def process_audio_file(self, file_path):
        # Leer el audio del archivo WAV
        wav = self.read_audio(file_path, sampling_rate=self.sampling_rate)

        # Normalizar el audio
        wav = normalize_audio(wav)

        # Obtener los segmentos de habla utilizando el modelo Silero VAD
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate)

        # Calcular la duración total de habla en los segmentos
        total_speech_duration = sum(segment['end'] - segment['start'] for segment in speech_timestamps)

        # Calcular la duración total del audio
        total_duration = len(wav) / self.sampling_rate

        # Calcular la proporción de habla en relación a la duración total
        is_speech = total_speech_duration / total_duration > 0.1

        if is_speech:
            self.process_speech_audio(wav)
        else:
            self.process_non_speech_audio(wav)

    def process_speech_audio(self, audio):
        # Convertir el audio a formato MP3
        mp3_audio = compress_audio(audio)

        # Enviar el audio a Whisper para transcripción
        result_whisper = self.send_audio_to_whisper(mp3_audio)

        if result_whisper is not None:
            # Realizar predicción con LongChain
            res_longchain = self.llm_chain.predict(human_input=result_whisper)

            if res_longchain is not None:
                # Sintetizar el texto con un servicio de síntesis de voz
                self.synthesize_audio(res_longchain)

    def process_non_speech_audio(self, audio):
        # Guardar el audio sin habla en un archivo temporal
        non_speech_file = self.create_temporary_wav_file()
        wave.write(non_speech_file, self.sampling_rate, audio)

        # Realizar acciones adicionales con el archivo de audio sin habla
        # (puede ser guardarlo en una base de datos, procesarlo, etc.)

    def send_audio_to_whisper(self, audio):
        segments, info = self.modelo.transcribe(audio, beam_size=5)
        print("Detected language '{}'' with probability {}".format(info.language, info.language_probability))
        response = ""

        for segment in segments:
            print("[{}s -> {}s] {}".format(segment.start, segment.end, segment.text))
            response += segment.text

        print("Whisper Transcription: {}".format(response))

        if response == "" or response == "Un poquito más." or response == "Subtítulos realizados por la comunidad de Amara.org" or response ==  "¡Gracias por ver el vídeo!":
            return None
        else:
            return response

    def synthesize_audio(self, text):
        url = "http://localhost:5000/synthesize"
        data = {"text": text}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, data=json.dumps(data), headers=headers)
        print(response)

    def signal_handler(self, sig, frame):
        print("Exiting program...")
        self.exit_event.set()


if __name__ == "__main__":
    server = AudioServer(port=9999)
    server.start_server()

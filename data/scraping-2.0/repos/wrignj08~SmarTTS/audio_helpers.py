import tempfile
from openai import OpenAI
import soundfile as sf
import simpleaudio as sa
from simpleaudio import WaveObject
from nltk.tokenize import sent_tokenize
from concurrent import futures
from tqdm.auto import tqdm
import librosa
import pyrubberband
from pydub import AudioSegment
from threading import Event
import time


def openai_tts(text: str, audio_file_path: str, hd=True) -> None:
    """
    Generates speech from text using OpenAI's TTS and saves it to a file.

    Args:
        text: The text to be converted to speech.
        audio_file_path: The file path where the audio will be saved.
    """
    if hd:
        model = "tts-1-hd"
    else:
        model = "tts-1"
    client = OpenAI()
    # Generate speech
    response = client.audio.speech.create(
        model=model,
        voice="alloy",
        input=text,
    )

    response.stream_to_file(audio_file_path)


def adjust_audio_speed(speed_factor: float, audio_file: str) -> None:
    """
    Adjusts the speed of an audio file.

    Args:
        speed_factor: Factor by which to adjust the speed.
        audio_file: Path to the audio file to be adjusted.
    """
    y, sr = librosa.load(audio_file, sr=None)
    y_stretched = pyrubberband.time_stretch(y, sr, speed_factor)
    sf.write(audio_file, y_stretched, sr, format="wav")


def create_audio_segment(text_chunk: str, speed_factor: float, use_hd) -> WaveObject:
    """
    Creates an audio segment from a text chunk with adjusted speed.

    Args:
        text_chunk: Text chunk to be converted to audio.
        speed_factor: Speed adjustment factor.

    Returns:
        A WaveObject representing the audio segment.
    """
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp3", mode="wb"
    ) as temp_file:
        audio_file_path = temp_file.name

        openai_tts(text_chunk, audio_file_path, use_hd)

        audio_segment = AudioSegment.from_file(str(audio_file_path))

        try:
            audio_segment.export(str(audio_file_path), format="wav")

            if speed_factor > 1:
                adjust_audio_speed(speed_factor, str(audio_file_path))

        except Exception as e:
            print(f"Error in audio manipulation: {e}")
            print("Audio too small to trim or speed up")

        wave_obj = sa.WaveObject.from_wave_file(str(audio_file_path))
        return wave_obj


def async_audio_generation(text: str, speed_factor: float, stop_event: Event) -> None:
    """
    Asynchronously generates and plays audio from text.

    Args:
        text: The text to be converted to speech.
        speed_factor: Speed factor for the audio.
        stop_event: An event to signal stopping the audio generation.
    """
    text_chunks = sent_tokenize(text)
    progress_bar = tqdm(total=len(text_chunks), desc="Playing audio")

    with futures.ThreadPoolExecutor(max_workers=2) as audio_gen_executor:
        # Store futures with their index and associated text chunk
        indexed_futures = {
            index: (
                audio_gen_executor.submit(
                    create_audio_segment, chunk, speed_factor, index == 0
                ),
                chunk,
            )
            for index, chunk in enumerate(text_chunks)
        }

        # Sort futures based on index before playback
        for index in sorted(indexed_futures.keys()):
            if stop_event.is_set():
                break

            future, chunk_text = indexed_futures[index]
            audio_obj = future.result()

            print(chunk_text)
            play_obj = audio_obj.play()

            while play_obj.is_playing():
                if stop_event.is_set():
                    play_obj.stop()
                    progress_bar.update(len(text_chunks) - index)
                    progress_bar.close()
                    break
                time.sleep(0.1)
            progress_bar.update(1)
        progress_bar.close()

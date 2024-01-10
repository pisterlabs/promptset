import pickle

import openai
from loguru import logger
from pydub import AudioSegment
from pydub.silence import detect_silence
from tqdm import tqdm

if __name__ == "__main__":
    audio = AudioSegment.from_file(r"GMT20230629-105932_Recording.m4a")

    logger.info(f"Audio duration: {len(audio) / 1000} seconds")

    with open(f"temp.mp3", "wb") as f:
        audio[:1000].export(f, format="mp3")
        # Warning: there be fuckery. m4a export doesn't work without strange magic, ogg is not supported by whisper atm
    logger.info("Codec works, format ok")

    # Parameters
    length_of_chunks = 15 * 60 * 1000  # in milliseconds, 15 mins
    min_silence_len = 1000  # minimum length of silence to be used for split. Adjust according to your needs
    silence_thresh = -40  # in dB, adjust according to your needs

    # Detect the silence intervals
    silences = detect_silence(audio, min_silence_len, silence_thresh, 1)

    logger.info(f"Found {len(silences)} silences")

    split_points = [0]
    for silence in silences:
        if silence[0] - split_points[-1] > length_of_chunks:
            split_points.append(silence[0])
    split_points.append(len(audio))

    logger.info(f"Split points: {split_points}")

    chunks = []
    for i in range(len(split_points) - 1):
        chunks.append(audio[split_points[i] : split_points[i + 1]])

    logger.info(f"Split into {len(chunks)} chunks")

    # Now you can save these chunks to files:
    for i, chunk in enumerate(chunks):
        with open(f"chunk_{i}.mp3", "wb") as f:
            chunk.export(f, format="mp3")

    logger.info("Chunks saved")

    joined_text = ""

    for i, chunk in tqdm(enumerate(chunks)):
        with open(f"chunk_{i}.mp3", "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f, language="ru")
        with open(f"transcript_{i}.pickle", "wb") as f:
            pickle.dump(transcript, f)

        joined_text += f"\n\nChunk {i}\n\n" + transcript.text

    with open("transcript.txt", "w", encoding="utf8") as f:
        f.write(joined_text.strip())

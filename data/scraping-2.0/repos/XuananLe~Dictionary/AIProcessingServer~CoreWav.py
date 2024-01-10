import asyncio
import base64
import concurrent
import WhisperFaster
import time
import app
import os
import openai
import whisper
import CoreTranslation


def wav_to_base64(wav_path):
    with open(wav_path, 'rb') as f:
        wav_data = f.read()
        base64_data = base64.b64encode(wav_data).decode('utf-8')
    return base64_data


def wav_to_text():
    start_time = time.time()
    audio_file = open("result.wav", "rb")
    result = openai.Audio.transcribe("whisper-1", audio_file)
    print("It took", time.time() - start_time, "seconds to transcribe the audio.")
    return result['text']


def base64_to_wav(base64_code):
    decoded_data = base64.b64decode(base64_code)
    wav_file = open('result.wav', 'wb')
    wav_file.write(decoded_data)
    wav_file.close()


def transcribe_wav():
    start_time = time.time()
    result = whisper.transcribe(app.main_model, "result.wav", fp16=False)
    end_time = time.time()
    transcribe_time = end_time - start_time
    print(f"Transcribe Time: {transcribe_time:.4f} seconds")
    return result["text"]


def wav_to_text2() -> str:
    start_time = time.time()

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            transcribe_future = executor.submit(transcribe_wav)
            text_result = transcribe_future.result()

        total_time = time.time() - start_time
        print(f"Total Time: {total_time:.4f} seconds")
        return text_result
    except whisper.WhisperException as e:
        print(f"WhisperException: ")
        return str(e)


def OpenAI_translate(text, language):
    prompt = f"Translate this {text} to {language}:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].text.strip()


async def main(base64_data: str):
    base64_to_wav(base64_data)
    text1 = WhisperFaster.wav_to_text()
    text2 = await CoreTranslation.TranslateManager.translate_word(text1, "en", "vi")
    print("Original text: ", text1)
    print("Translated text: ", text2)

    return text1, text2


if __name__ == '__main__':
    print("Start")
    asyncio.run(main(wav_to_base64("Client.wav")))

import base64
import logging
import os
from typing import AsyncIterable

from azure.ai.textanalytics import TextAnalyticsClient
from azure.cognitiveservices import speech
from azure.core.credentials import AzureKeyCredential
import openai

openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_URL")
openai.api_version = "2023-05-15"

OPENAI_CHATGPT_DEPLOYMENT = os.getenv("OPENAI_CHATGPT_DEPLOYMENT")
OPENAI_GPT_DEPLOYMENT = os.getenv("OPENAI_GPT_DEPLOYMENT")

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")

LLM_DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.0"))

available_voices: list[speech.VoiceInfo] = speech.SpeechSynthesizer(
    speech_config=speech.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION), audio_config=None
).get_voices_async().get().voices

# For Azure at-start recognition we can only use 4. We could up to 10 using continuous recognition
# This may become irrelevant if we use frontend STT (Webkit Speech API) instead of Azure
speech_to_text_languages = ["en-GB", "de-DE", "es-ES", "fr-FR"]


def perform_chat_completion(
    history: list[dict], prompt: str, temperature: float = LLM_DEFAULT_TEMPERATURE, **kwargs
) -> dict[str, str]:
    messages = history + [{"role": "user", "content": prompt}]

    chat_completion = openai.ChatCompletion.create(
        deployment_id=OPENAI_CHATGPT_DEPLOYMENT,
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        **kwargs,
    )

    output = chat_completion.choices[0].message.content

    return {
        "output": output
    }


async def perform_chat_completion_async(
    history: list[dict], prompt: str, temperature: float = LLM_DEFAULT_TEMPERATURE, **kwargs
) -> dict[str, str]:
    messages = history + [{"role": "user", "content": prompt}]

    chat_completion = await openai.ChatCompletion.acreate(
        deployment_id=OPENAI_CHATGPT_DEPLOYMENT,
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        **kwargs,
    )

    output = chat_completion.choices[0].message.content

    return {
        "output": output
    }


async def perform_chat_completion_streaming(
    history: list[dict], prompt: str, temperature: float = LLM_DEFAULT_TEMPERATURE, **kwargs
) -> AsyncIterable[dict[str, str]]:
    messages = history + [{"role": "user", "content": prompt}]

    chat_completion = await openai.ChatCompletion.acreate(
        deployment_id=OPENAI_CHATGPT_DEPLOYMENT,
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        stream=True,
        **kwargs,
    )

    async for chunk in chat_completion:
        yield {
            "output": chunk["choices"][0]["delta"]
        }


def perform_speech_to_text(filename: str = None, content: bytes = None) -> dict:
    speech_config = speech.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"

    if filename:
        audio_config = speech.audio.AudioConfig(filename=filename)
    elif content:
        # TODO: is this right? who knows
        stream = speech.audio.PushAudioInputStream()
        stream.write(content)
        audio_config = speech.audio.AudioConfig(stream=stream)
    else:
        raise ValueError("Must provide filename or content")

    detect_lang_config = speech.languageconfig.AutoDetectSourceLanguageConfig(languages=speech_to_text_languages)

    recognizer = speech.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config, auto_detect_source_language_config=detect_lang_config
    )

    result = recognizer.recognize_once_async().get()

    if result.reason == speech.ResultReason.NoMatch:
        return {
            "output": ""
        }
    if result.reason == speech.ResultReason.Canceled:
        error_message = "Error in speech transcription"
        cancellation_details = result.cancellation_details
        if cancellation_details.reason == speech.CancellationReason.Error:
            error_message = "Error details: {}".format(cancellation_details.error_details)
        raise Exception(error_message)

    return {
        "output": result.text
    }


def perform_language_recognition(text: str) -> str:
    credential = AzureKeyCredential(AZURE_LANGUAGE_KEY)
    client = TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=credential)
    response = client.detect_language(documents=[text])[0]
    language_obj = response.primary_language
    return language_obj.iso6391_name


def perform_text_to_speech(text: str, lang: str = "auto") -> dict:
    if lang == "auto":
        try:
            lang = perform_language_recognition(text)
            if "_" in lang:
                lang = lang.split("_")[0]
        except Exception:
            logging.warning("Exception when recognising language, defaulting to 'en'...")
            lang = "en"

    speech_config = speech.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.set_speech_synthesis_output_format(speech.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3)

    for voice in available_voices:
        if lang in voice.locale:
            # this isn't a good way of doing it but there are many voices per lang so it's not clear what a better way is
            speech_config.speech_synthesis_voice_name = voice.name
            break

    synthesizer = speech.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speech.ResultReason.Canceled:
        error_message = "Error in speech synthesis"
        cancellation_details = result.cancellation_details
        if cancellation_details.reason == speech.CancellationReason.Error:
            if cancellation_details.error_details:
                error_message = "Error details: {}".format(cancellation_details.error_details)
        raise Exception(error_message)

    audio_data = bytearray()
    buffer = bytes(16000)
    stream = speech.AudioDataStream(result)

    while (filled_size := stream.read_data(buffer)) > 0:
        audio_data.extend(buffer[:filled_size])

    return {
        "output": base64.b64encode(audio_data).decode()
    }



import openai
import os
from stt_transcription import STT
from dotenv import load_dotenv
import logging
from time import perf_counter
from llama_cpp import Llama
import json

load_dotenv()
logger = logging.getLogger(__name__)

def openai_feedback():
    """
    This function is used to engage in feedback with the user.
    The user will be asked to provide feedback on the generated transcript.
    """
    openai.api_key = os.getenv('OPEN_AI_API_KEY')
    transcript = STT.openai_file_transcription(
        audio_file="audio_data/audio_sample_2.wav",
        prompt_guidance='Tienes que detectar quien habla en el audio.'
    )
    logger.info('OPENAI API is running for feedback...\n\n')
    start_time = perf_counter()
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "Eres un asistente que proporciona información sobre un archivo de audio transcrito."
                           "Basándote en la situación dada, detecta quién es el entrevistador y quién es el entrevistado."
                           "¿Cómo ayudarias al entrivistado a mejorar su entrevista?"
            },
            {"role": "user", "content": f"Esta es la transcripción: {transcript}"},

        ]
    )
    logger.info(f'Time taken by OPENAI GPT-3: {perf_counter() - start_time} seconds.')
    return completion.choices[0]['message']['content']

def ask_llama(
        model_path: os.PathLike,
        prompt: str
) -> str:
    llm = Llama(model_path)
    start_time = perf_counter()
    output = llm(
        prompt,
        max_tokens=100,
        stop=["Q:", "\n"],
        echo=True
    )

    logger.info(f'Time taken by Llama: {perf_counter() - start_time} seconds.')
    return output['choices'][0]['text']


if __name__ == '__main__':
    # print(ask_llama(
    #     model_path='models/llama-13b/llama-2-13b.Q4_K_S.gguf',
    #     prompt='Question: How to run faclon 180B in my local machine?\n'
    #            'Answer: '
    # ))
    from langchain.llms import CTransformers
    import yaml

    # load yaml config file:
    with open('../config/config_llama.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    print(config)


    llm = CTransformers(model='models/llama-13b/llama-2-13b.Q4_K_M.gguf')

    print(llm('AI is going to'))








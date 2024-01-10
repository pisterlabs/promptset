from typing import Tuple, Optional, List, Dict
import time
import library.errors as errors
import os
import json
import requests as req
import logging
from elevenlabs import generate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.llms import OpenAIChat
import openai
import openai.error
from sapling import SaplingClient

logger = logging.getLogger('Service')

PT_BR = 'brazilian portuguese'
EN_US = 'english'
LANGUAGES = [PT_BR, EN_US]

def AskChatGPT(messages: List[dict], model: Optional[str] = os.getenv('CHAT_GPT_3_MODEL'), temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = 2000, useChain: Optional[bool] = False) -> Tuple[str, str]:

    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')

    if model == os.getenv('CHAT_GPT_4_MODEL'):
        openai.api_key = os.getenv('GPT_4_API_KEY')
        openai.organization = os.getenv('GPT_4_ORGANIZATION')

    lastError = None

    for msg in messages:
        if 'content' not in msg or 'role' not in msg:
            raise errors.BadRequest('invalid messages list. missing content or role fields')

    for i in range(0, 3):
        try:
            if useChain:
                baseMessages = []
                for msg in messages:
                    if msg['role'] == 'system':
                        schemaMsg = SystemMessage(content=msg['content'])
                    elif msg['role'] == 'user':
                        schemaMsg = HumanMessage(content=msg['content'])
                    else:
                        schemaMsg = AIMessage(content=msg['content'])

                    baseMessages.append(schemaMsg)

                prompt = ChatPromptTemplate(input_variables=[], messages=baseMessages)

                llm = OpenAIChat(
                    temperature=temperature,
                    model_name=model,
                    max_tokens=max_tokens,
                ) # Always use chatGPT 3.5-turbo and default org

                chain = LLMChain(llm=llm, prompt=prompt)
                runContent = chain.run()

                response = {'choices': [{'message': {'content': runContent, 'role': 'assistant'}}]}

            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=1,
                    max_tokens=max_tokens,
                    stream=False,
                )

            if not useChain and response['choices'][0]['finish_reason'] == 'length':
                raise errors.OpenAiRequestError() # Force retry when the response comes missing content

        except openai.error.RateLimitError as e:
            logger.warning('Rate limit atingido na OpenAI! {}'.format(e))
            raise errors.OpenAiRateLimitError()

        except openai.error.APIConnectionError as e:
            logger.error('Problema de conexão com a API da OpenAI: {}'.format(e))
            raise errors.InternalError('ocorreu um problema de conexão com a OpenAI')

        except openai.error.InvalidRequestError as e:
            raise errors.BadRequest('requisição inválida para a OpenAI; {}'.format(e))

        except openai.error.AuthenticationError as e:
            logger.error('API token para OpenAI inválido, expirado ou revogado! {}'.format(e))
            raise errors.InternalError('ocorreu uma falha na autenticação para uso da API da OpenAI')

        except (openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError, errors.OpenAiRequestError) as e:
            lastError = e
            continue

        except Exception as e:
            logger.error(e)
            raise errors.InternalError('aconteceu algum problema ao tentar criar um chat completion: {}'.format(e))

        else:
            break

    else:
        logger.error('Falha na comunicação com a API da OpenAI! Último erro retornado: {}'.format(lastError))
        raise errors.OpenAiCommunicationError()

    if 'error' in response:
        logger.error('OpenAI retornou erro na request não tratado: {}'.format(response['error']))
        raise errors.OpenAiRequestError()

    elif 'id' not in response or not ('message' in response['choices'][0] and 'content' in response['choices'][0]['message']):
        raise errors.OpenAiRequestError()

    return response['choices'][0]['message']['content'], response['choices'][0]['message']['role']

def Translate(text: str) -> str:
    prompt = """
        Detect the language of the text delimited by triple backticks, then do one of the following options:
        1 - If the text is in English language, translate it to Brazilian portuguese.
        2 - If the text is in Brazilian portuguese language, translate it to English.
        ```{}```
    """

    try:
        content, _ = AskChatGPT(
            messages=[
                {'role': 'user', 'content': prompt.format('Música brasileira')},
                {'role': 'assistant', 'content': 'Brazilian music'},
                {'role': 'user', 'content': prompt.format('Amigos em comum')},
                {'role': 'assistant', 'content': 'Mutual friends'},
                {'role': 'user', 'content': prompt.format('Favorite food')},
                {'role': 'assistant', 'content': 'Comida favorita'},
                {'role': 'user', 'content': prompt.format('Job interviews')},
                {'role': 'assistant', 'content': 'Entrevistas de emprego'},
                {'role': 'user', 'content': prompt.format("That's great! It's always nice to explore different music styles and genres. Brazilian music has a lot to offer.")},
                {'role': 'assistant', 'content': 'Isso é ótimo! É sempre bom explorar diferentes estilos e gêneros musicais. A música brasileira tem muito a oferecer.'},
                {'role': 'user', 'content': prompt.format("If you're into indie songs, you might enjoy some Brazilian indie artists as well!")},
                {'role': 'assistant', 'content': 'Se você gosta de músicas indie, pode gostar de alguns artistas indie brasileiros também!'},
                {'role': 'user', 'content': prompt.format(text)}
            ],
            temperature=0,
        )

    except:
        raise

    return content

def GetTextCorrections(text: str) -> List[dict]:
    client = SaplingClient(os.getenv('SAPLING_API_KEY'))
    corrections = []

    try:
        edits = client.edits(
            text=text,
            lang='en',
            variety='us-variety'
        )

        for edit in edits:
            start = edit['start']
            end = edit['end']
            correction = edit['replacement']

            corrections.append({'from': text[start:end], 'to': correction})

    except Exception as e:
        logger.error(e)
        pass

    return corrections

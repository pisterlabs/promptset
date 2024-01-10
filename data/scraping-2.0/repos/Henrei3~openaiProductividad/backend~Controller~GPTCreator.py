from backend.Model.RequestModel import ChatGPTRequestModel, AudioGPTRequestModel
from backend.Model.RequestModel import EmbeddingRequestModel, OpenAIModelInterface
from backend.Controller.CalculoPrecios import AudioPriceCalculation, EmbeddingsPriceCalculation, Currency
from backend.Model.DB.recordingsDB import Recording
from abc import ABC, abstractmethod
from decouple import config
from typing import Union
from pydub import AudioSegment
import json
import openai
import os


class OpenAIProxy(ABC):

    @classmethod
    def check_access(cls, openai_model: OpenAIModelInterface, authorized: bool):
        """ This is the main method of the proxy class. It returns the response of the request in the case that
         it already exists, or it organizes the execution of said request and saves the response in the database.
         Additionally, if authorized is set to False you will get the cost of the request instead of actually
         executing it"""
        response: Union[str, None] = openai_model.get_response()
        if response is None:
            operation_result = cls.operation(openai_model, authorized)
            if type(operation_result) is not float:
                openai_model.set_response(operation_result)
                return openai_model.get_response()
            else:
                return operation_result
        else:
            return response

    @staticmethod
    @abstractmethod
    def operation(openai_model, authorized: bool):
        """ This method funnels all the different operations once check access has allowed their execution """
        pass


class OpenAIProxyAudio(OpenAIProxy):
    """ This method implements the abstract class OpenAiProxy to override the operation method for the
    Audio Module of the application"""
    @staticmethod
    def operation(openai_model: AudioGPTRequestModel, authorized: bool):
        if authorized:
            return OpenAIAudioRequest.execute_request(openai_model)
        else:
            return OpenAIAudioRequest.calculate_price(openai_model)


class OpenAIProxyEmbeddings(OpenAIProxy):
    """ This method implements the abstract class OpenAIProxy to override the operation method for the
     Embeddings Module of the application
     """
    @staticmethod
    def operation(openai_model, authorized: bool):
        if authorized:
            return OpenAIEmbeddingRequest.execute_request(openai_model)
        else:
            return OpenAIEmbeddingRequest.calculate_price(openai_model)


class OpenAiRequestMeta(type):
    """ This class specifies the metadata of the OpenAIRequestInterface
     It is one options to make an Interface in Python"""
    def __instancecheck__(self, instance):
        return self.__subclasscheck__(type(instance))

    def __subclasscheck__(self, subclass):
        return (hasattr(subclass, 'execute_request') and
                callable(subclass.execute_request) and
                hasattr(subclass, 'calculate_price') and
                callable(subclass.calculate_price))


class OpenAIRequestInterface(metaclass=OpenAiRequestMeta):
    """ Interface that groups together all the OpenAI Requests """
    pass


class OpenAIAudioRequest:
    """  This class makes a call to the OpenAI api to make use of whisper-1  """

    APIKEY = config('whisper')

    @classmethod
    def execute_request(cls, gpt_model: AudioGPTRequestModel) -> str:
        """
        This method will check if the audio_size doesn't exceed 24 m since the OpenAPI can transform audios up to 25 mb.
        If the audio exceeds 25 mb, we will split it in two and pass it throw the api.
        """
        audio_path: str
        # We segment the audio in case it is too big for the API to process
        if gpt_model.size >= 24:
            audio_segment = AudioSegment.from_wav(gpt_model.get_audio_path())
            duration_in_milliseconds_split_in_two = (audio_segment.duration_seconds * 1000) / 2

            response: str = ""
            for i in range(0, 2):
                segmented_audio = audio_segment[
                                  duration_in_milliseconds_split_in_two*i:duration_in_milliseconds_split_in_two*(i+1)
                                  ]
                segmented_audio.export("segmented_audio.wav", format="wav")
                response += cls._api_call("segmented_audio.wav", "")

            os.remove("segmented_audio.wav")
            return json.loads(response)["text"]
        # We do a regular call in case it is a moderate size
        else:
            return cls._api_call(gpt_model.get_audio_path(), gpt_model.get_prompt())

    @classmethod
    def _api_call(cls, path: str, prompt: str):
        with open(path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                api_key=cls.APIKEY,
                model="whisper-1",
                file=audio_file,
                prompt=prompt
            )
        print(response)
        return response["text"]

    @classmethod
    def diarize(cls, audio_path):
        pass
    @staticmethod
    def calculate_price(gpt_request: AudioGPTRequestModel):
        return AudioPriceCalculation.audio_request_price_calculation(gpt_request, Currency.AudioPricing.USD)


class OpenAIChatRequest:
    """ This class uses the most known model of OpenAI turbo-3.5 """
    @staticmethod
    def execute_request(gpt_model: ChatGPTRequestModel) -> str:

        APIKEY = config('whisper')
        messages = gpt_model.get_message()

        response = openai.ChatCompletion.create(
            api_key=APIKEY,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000,
            messages=messages
        )

        print(response)
        return response['choices'][0]['message']['content']

    @staticmethod
    def message_parser(raw_message, system):
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": raw_message}
        ]
        return message

    @staticmethod
    def calculate_price(gpt_model: ChatGPTRequestModel):
        print("Not Yet Implemented")


class OpenAIEmbeddingRequest:
    """ This class calls the ada embeddings model to create embeddings of certain texts """

    @staticmethod
    def execute_request(gpt_model: EmbeddingRequestModel) -> str:
        openai.api_key = config("embeddings")
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=gpt_model.get_text()
        )

        print(response)
        return response["data"][0]["embedding"]

    @staticmethod
    def calculate_price(gpt_model: EmbeddingRequestModel):
        return EmbeddingsPriceCalculation.embeddings_calculation(gpt_model, Currency.EmbeddingsPricing.USD)




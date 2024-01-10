import json
from time import time
from pdx.logger import logger
from pdx.models.model import Model
from pdx.prompt.prompt_session import PromptSession
from pdx.models.openai.client import OpenAIClient
from pdx.models.metadata import ModelResponse, ResponseMetadata
from pdx.models.utils.image import format_response


class AudioTranscriptionModel(Model):
    def __init__(self,
                 api_key: str,
                 **kwargs,
                 ):

        self._api_url = "v1/audio/transcriptions"

        self._provider = "openai"
        self._client = OpenAIClient(api_key)
        # self._client.
        self._model = kwargs.get('model', 'whisper-1')
        # response_format Optional[str] = 'json' options: json, text, srt, verbose_json, or vtt
        self._response_format = kwargs.get('response_format', 'text')
        # temperature Optional[float] = 0.0 options: 0.0 - 1.0
        self._temperature = kwargs.get('temperature', 0.0)
        # language  Optional[str] = 'en' options: ISO-639-1 https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        self._language = kwargs.get('language', 'en')

        self._retries = kwargs.get('retries', 2)

    def _preprocess(self, prompt: PromptSession) -> dict:
        request_params = {
            "model": self._model,
            "temperature": self._temperature,
            "language": self._language,
            "response_format": self._response_format,
        }
        _files: list = prompt.audio_prompt()
        if len(_files) > 1:
            logger.echo('Only one audio prompt supported at the moment.')
        _prompt = prompt.text_prompt({})
        request_params['files'] = {
            "file": (_files[0][0], _files[0][1], "application/octet-stream")}
        request_params['prompt'] = _prompt

        return request_params

    def _postprocess(self, response: dict, request_params: dict, request_time: float) -> ModelResponse:
        _prompt = request_params.pop('prompt', None)
        _files = request_params.pop('files', None)

        if request_params['response_format'] in {'json', 'verbose_json'}:
            _response = json.loads(response)
            _data = _response['text']
        else:
            _response = response
            _data = response

        response_metadata = ResponseMetadata(
            model=request_params['model'],
            stop='transcribe_completed',
            stop_reason='transcribe_completed',
            latency=request_time)

        _data_type = f'text_string'

        model_response = ModelResponse(
            metadata=response_metadata,
            request_params=request_params,
            data=_data,
            data_type=_data_type
        )

        return model_response

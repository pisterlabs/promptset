from os import environ, getenv
from os.path import join
from typing import Dict, Iterator, List, TypeVar, Union

from box.box import Box
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from openai._streaming import Stream
from openai.types.chat import ChatCompletion

from utils import (
    detect_is_work_device,
    get_src_dir_path,
    num_tokens_from_messages,
    num_tokens_from_string,
    read_yaml,
)


class OAIClient:
    # Default params (will be overwritten if param is specified in config file!)
    MODEL: str = "gpt-3.5-turbo-1106"
    TEMPERATURE: float = 0.2
    SEED: int = 12345
    TOP_P: float = 1.0
    _POSTPROCESS_STREAM: bool = False

    # Session state params
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    pricing_cost: float = 0.0
    num_calls: int = 0

    # Hardcoded
    # Cost of model calls (per 1000 tokens) (input, output costs)
    cost_mapping = {
        "gpt-3.5-turbo-1106": (0.0010, 0.0020),
        "gpt-3.5-turbo-0613": (0.0015, 0.0020),
        "gpt-3.5-turbo-16k-0613": (0.0030, 0.0040),
    }

    def __init__(
        self,
        config_file: Box = None,
        client: OpenAI = None,
        **kwargs,
    ) -> None:
        """
        Models list: https://platform.openai.com/docs/models
        Models pricing: https://openai.com/pricing // https://platform.openai.com/docs/deprecations/2023-11-06-chat-model-updates
        Nice costcalculating library: https://www.reddit.com/r/Python/comments/12lec2s/openai_pricing_logger_a_python_package_to_easily/
        """
        self._parse_config(config_path=config_file)
        self.client = client if client else self._init_client(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.query(*args, **kwargs)

    def query(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        seed: int = None,
        top_p: float = None,
        max_tokens: int = None,
        return_raw_response: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, ChatCompletion]:
        # Defaults
        model = model if model is not None else self.MODEL
        temperature = temperature if temperature is not None else self.TEMPERATURE
        seed = seed if seed is not None else self.SEED
        top_p = top_p if top_p is not None else self.TOP_P

        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )
        # Update state
        # TODO: Put in postprocessing step
        self.num_calls += 1
        if stream and not isinstance(response, Stream):
            raise ValueError("Response is streamed, but not an OpenAI.Stream object!")
        if stream and self._POSTPROCESS_STREAM:
            # TODO: Implement postprocessing for stream (MODIFY STREAM OBJECT to count tokens etc.)
            # self._postprocess_stream(messages=messages, model=model, response=response)
            pass
        elif not stream:
            self._postprocess(messages=messages, model=model, response=response)
        else:
            print("WARNING: Streaming response so skipping postprocessing step!")

        if return_raw_response:
            return response
        else:
            return response.choices[0].message.content

    def _postprocess(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response: Union[ChatCompletion, str],
    ) -> None:
        """Postprocess response."""
        self._track_tokens_used_in_call(messages=messages, model=model, response=response)
        self._track_cost_of_call(model=model)

    def _postprocess_stream(self, stream: Stream) -> None:
        """
        Postprocess response as stream.

        Replace method of existing object: https://stackoverflow.com/a/2982/10002593
        (USE ATTRIBUTE INSTEAD!) Passing parameters to the __iter__ method: https://stackoverflow.com/a/54395995/10002593
        """

        def newiterfunc(self) -> Iterator[TypeVar("_T")]:
            """
            Taken from: https://github.com/openai/openai-python/blob/main/src/openai/_streaming.py#L21
            """
            for item in self._iterator:
                yield item
            # TODO: Do something with self.OAI_CLIENT

        stream.OAI_CLIENT = self
        stream.__iter__ = newiterfunc.__get__(stream)

    def _track_cost_of_call(self, model: str) -> None:
        """
        Calculate cost of call.
        """
        input_token_cost, output_token_cost = self.cost_mapping[model]  # per 1000 token
        self.pricing_cost += input_token_cost * (
            self.input_tokens_used / 1000
        ) + output_token_cost * (self.output_tokens_used / 1000)

    def _track_tokens_used_in_call(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response: Union[ChatCompletion, str],
        manual_calculation: bool = False,
    ):
        """Calculate number of tokens used in call."""
        response_content = (
            response if isinstance(response, str) else response.choices[0].message.content
        )
        manual_calculation = True if isinstance(response, str) else manual_calculation
        self.output_tokens_used += (
            num_tokens_from_string(string=response_content, model=model)
            if manual_calculation
            else response.usage.completion_tokens
        )
        self.input_tokens_used += (
            num_tokens_from_messages(messages=messages, model=model)
            if manual_calculation
            else response.usage.prompt_tokens
        )

    def _test_connection(self, print_output: bool = False) -> bool:
        try:
            response = self.client.query(
                messages=[
                    {
                        "role": "user",
                        "content": "say hi",
                    },
                ],
                return_raw_response=True,
            )
            if print_output:
                print(response)
                print(response.choices[0].message.content)
            return True
        except Exception as e:
            print(e)
            return False

    def _close_client(self) -> None:
        """Close OpenAI client."""
        if not self.client.is_closed:
            self.client.close()

    def _parse_config(self, config_path: str) -> None:
        """Parse config file from input path."""
        self.config = read_yaml(input_path=config_path) if config_path is not None else None
        if config_path is not None:
            for key, value in self.config.items():
                if key in OAIClient.__dict__.keys():
                    setattr(self, key, value)

    def _init_client(
        self,
        **kwargs,
    ) -> OpenAI:
        """Initialize OpenAI client."""
        try:
            load_dotenv()
        except Exception as e:
            pass

        # Take input, otherwise config, otherwise dotenv
        relevant_params = [
            ("api_key", "OPENAI_API_KEY"),
            ("api_version", "OPENAI_API_VERSION"),
            ("azure_endpoint", "AZURE_OPENAI_ENDPOINT"),
        ]
        kwargs_subset = dict()
        for input_param, config_param in relevant_params:
            if input_param in kwargs.keys():
                kwargs_subset.update({input_param: kwargs[input_param]})
            elif hasattr(self, config_param):
                kwargs_subset.update({input_param: self.config_param})
            elif config_param in environ:
                kwargs_subset.update({input_param: getenv(config_param)})

        if detect_is_work_device():
            client = AzureOpenAI
        else:
            client = OpenAI
        return client(**kwargs_subset)


if __name__ == "__main__":
    CONFIG_FILE_PATH = join(get_src_dir_path(), "config.yaml")
    client = OAIClient(config_file=CONFIG_FILE_PATH)
    _ = client._test_connection(print_output=True)
    print(client.pricing_cost)
    print(client.input_tokens_used, client.output_tokens_used)

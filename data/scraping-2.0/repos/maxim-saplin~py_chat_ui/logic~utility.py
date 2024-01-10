from logic import env_vars
from logic.user_state import Model


def create_client(model: Model):
    from openai import OpenAI, AzureOpenAI
    if model.api_type == env_vars.ApiTypeOptions.AZURE:
        return AzureOpenAI(api_key=model.api_key,
                           azure_endpoint=model.api_base,
                           api_version=model.api_version,
                           azure_deployment=model.model_or_deployment_name)
    elif model.api_type == env_vars.ApiTypeOptions.FAKE:
        return FakeOpenAI()
    else:
        base_url = model.api_base if model.api_base else None
        return OpenAI(api_key=model.api_key, base_url=base_url)


encoding = None


def num_tokens_from_messages(messages: list[dict]) -> int:
    """Return the number of tokens used by a list of messages."""
    import tiktoken
    global encoding

    if not encoding:
        encoding = tiktoken.get_encoding("cl100k_base")

    if not messages:
        return 0

    tokens_per_message: int = 4  # every message follows <|im_start|>{role/name}\n{content}<|end|>\n
    tokens_per_name: int = -1  # if there's a name, the role is omitted

    num_tokens: int = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|im_start|>assistant<|im_sep|>
    return num_tokens


class FakeOpenAI:
    def __init__(self):
        self.chat = self.Chat()

    class Chat:
        def __init__(self):
            self.completions = self.Completions()

        class Completions:
            def create(self, *args, **kwargs):
                yield self.ChatCompletionChunk()

            class ChatCompletionChunk:
                def __init__(self):
                    self.choices = [self.Choice()]

                class Choice:
                    def __init__(self):
                        self.content = 'Hello world!'

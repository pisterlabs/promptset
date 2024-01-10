import openai
import tiktoken

from Const import OPENAI_API_KEY


class OpenAIServer(object):
    def __init__(self, model="text-davinci-003", api_base=None, is_llama=False):
        openai.api_key = OPENAI_API_KEY
        if api_base is not None:
            assert is_llama is True, "api_base is only used for llama"
            openai.api_base = api_base
            openai.api_key = "EMPTY"
        self.is_llama = is_llama
        self.model = model
        self.mode = "completion" if "text-davinci" in self.model else "chat"
        pass

    def is_chat_model(self):
        return self.mode == "chat"

    def send(self, workload, logprobs=5, **kwargs):
        if "text-davinci" in self.model or self.mode == "completion":
            response = self.send_completion(workload, logprobs)
        elif "turbo" in self.model or self.mode == "chat":
            response = self.send_chat(workload, **kwargs)
        else:
            if self.mode == "accounting":
                response = self.send_accounting(workload, **kwargs)
            raise NotImplementedError("unknown model type")
        return response

    def send_accounting(self, workload, **kwargs):
        def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
            """Returns the number of tokens used by a list of messages."""
            # official codes from OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # print("Warning: model not found. Using cl100k_base encoding.")
                encoding = tiktoken.get_encoding("cl100k_base")
            if model == "gpt-3.5-turbo":
                # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
                return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
            elif model == "gpt-4":
                # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
                return num_tokens_from_messages(messages, model="gpt-4-0314")
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif model == "gpt-4-0314":
                tokens_per_message = 3
                tokens_per_name = 1
            else:
                raise NotImplementedError(
                    f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        if "turbo" in self.model or "gpt-4" in self.model:
            return num_tokens_from_messages(workload, self.model)
        else:
            try:
                encoding = tiktoken.encoding_for_model(self.model)
                return len(encoding.encode(workload))
            except:
                return len(workload)

    def send_chat(self, workload, **kwargs):
        # for chat-system, we can not have it output logprobs
        model = kwargs.get("model", self.model)
        try:
            assert "turbo" in model or self.is_llama, "mis-specified model type to do chat completion"
        except AssertionError as e:
            print(e)
            model = "gpt-3.5-turbo"
        message_workload = self.prepare_chat_workload(workload, **kwargs)
        # Following OpenAI default setting. There might be other settings that are better but we leave it for future work
        response = openai.ChatCompletion.create(
            model=model,
            messages=message_workload,
            max_tokens=kwargs.get("max_tokens", 1024),
            n=kwargs.get('n', 1),
            top_p=kwargs.get('top_p', 0.95)
        )

        return response

    def prepare_chat_workload(self, workload, **kwargs):
        if type(workload) == str:
            if "system" in kwargs:
                system_msg = kwargs['system']
            else:
                system_msg = "You are a helpful assistant."
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": workload}
            ]
            return messages
        else:
            assert type(workload) == list, "workload must be a string or a list of dicts"
            if "system" in kwargs:
                system_msg = kwargs['system']
            else:
                system_msg = "You are a helpful assistant."
            messages = [
                {"role": "system", "content": system_msg},
            ]
            return messages + workload

    def send_completion(self, workload, logprobs=5):
        response = openai.Completion.create(
            model=self.model,
            prompt=workload,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=logprobs
        )
        return response

    def demo(self):
        response = openai.Completion.create(
            model=self.model,
            prompt="Instructions: As an agent, you need to find the way to go out of this quest. Currently, there is a box in front of you and there is a key inside. You can use this key to open the door and finish this quest. There are 5 boxes and 5 keys here. Boxes are identified as Fe-X and Keys are identified as Mxx-X.  GSiBamFx(Fe-4)=True means that Fe-4 has been opened. Vt(Mxx-0)=True means that Mxx-0 has been obtained. GSiBamFx(Fe-4)=False means that Fe-4 has not been opened. Vt(Mxx-0)=False means that Mxx-0 has not been obtained. Step-1: Open Fe-1 and retrieve Mxx-2\nQuestion: GSiBamFx(Fe-0)=?GSiBamFx(Fe-1)=?GSiBamFx(Fe-2)=?GSiBamFx(Fe-3)=?GSiBamFx(Fe-4)=?Vt(Mxx-0)=?Vt(Mxx-1)=?Vt(Mxx-2)=?Vt(Mxx-3)=?Vt(Mxx-4)=?\nAnswer: ",
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=5
        )
        return response

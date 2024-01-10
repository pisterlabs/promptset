from transformers import pipeline, Conversation

import openai


class Responder:
    def __call__(self, messages: list[dict], stream: bool):
        raise NotImplementedError


class OpenAIResponder(Responder):
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name

    def __call__(self, messages: list[dict], stream: bool):
        if stream:
            for resp in openai.ChatCompletion.create(model=self.model_name, messages=messages, stream=True):
                if "content" in resp["choices"][0]["delta"].keys():
                    yield resp["choices"][0]["delta"]["content"]
        else:
            return openai.ChatCompletion.create(model=self.model_name, messages=messages)["choices"][0]["message"]["content"]


class HuggingFaceResponder(Responder):
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.responder = pipeline("conversational", model_name)

    def __call__(self, messages: list[dict]):
        # TODO: Support streaming
        return self.responder(Conversation(messages.copy())).generated_responses[-1]


class ChatBot:
    def __init__(self, responder: Responder, system_message: str):
        self.messages = [{"role": "system", "content": system_message}]
        self.responder = responder

    def respond(self, message: str):
        self.messages.append({"role": "user", "content": message})
        response = self.responder(self.messages, stream=False)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def respond_stream(self, message: str):
        self.messages.append({"role": "user", "content": message})
        segments = []
        segment = []
        for token in self.responder(self.messages, stream=True):
            segment.append(token)
            # TODO: Use exponentially increasing max and min lengths with check for punctuation, to minimize latency
            #  and the occurence of pauses in the middle of subsentences
            if len(segment) >= 10:  # token in {"\n", ".", ",", ":", ";", "!", "?"} (or maybe string.punctuation)
                segments.append("".join(segment))
                segment = []
                yield segments[-1]
        segments.append("".join(segment))
        self.messages.append({"role": "assistant", "content": "".join(segments)})
        yield segments[-1]


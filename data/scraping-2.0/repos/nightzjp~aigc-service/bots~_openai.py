import os
import openai


class OpenAIClient:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = api_key

    def completion(
        self,
        prompt,
        model="text-davinci-003",
        stream=False,
        temperature=0.7,
        top_p=1,
        **kwargs
    ):
        return openai.Completion.create(
            model=model,
            prompt=prompt,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def chat_completion(
        self,
        messages,
        model="gpt-3.5-turbo-0301",
        stream=False,
        temperature=0.7,
        top_p=1,
        **kwargs
    ):
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )


if __name__ == "__main__":
    openai_client = OpenAIClient()

    # res = openai_client.completion(
    #     "This is a test",
    # )
    # print(res)
    messages = [
        {"role": "system", "content": "你是一个测试员，将回应我的测试要求"},
        {"role": "user", "content": "请回复啊啊啊成功了"},
    ]
    stream_res = openai_client.chat_completion(messages, stream=True)
    for r in stream_res:
        print(r)
        print("\n")

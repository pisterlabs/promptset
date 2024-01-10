import openai
if __name__ == "__main__":
    openai.api_base = "http://192.168.0.103:30021/v1"
    #openai.api_base = "http://10.244.36.135:8081/v1"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
        model="chatglm2-6b",
        messages=[
            {"role": "user", "content": "what is ai"}
        ],
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)

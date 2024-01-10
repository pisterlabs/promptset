import argparse
import openai
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the OpenAI API")
    parser.add_argument("--ip", required=True, help="IP address and port for the API server")
    parser.add_argument("--prompt",default="What is AI", help="Prompt for the conversation")
    args = parser.parse_args()

    api_base_url = f"http://{args.ip}/v1"
    openai.api_base = api_base_url #"http://172.16.3.21:30021/v1"
    #openai.api_base = "http://10.110.138.201:30021/v1"
    #openai.api_base = "http://10.244.36.135:8081/v1"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
        model="chatglm2-6b",
        messages=[
            {"role": "user", "content": args.prompt}
        ],
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)

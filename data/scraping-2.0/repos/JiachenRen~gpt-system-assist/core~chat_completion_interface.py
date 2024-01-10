import openai


class ChatCompletionInterface:

    def __init__(self, model):
        self.model = model

    # Get chat completion response from GPT
    def get_chat_completion_response(self, messages, functions):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=functions,
            temperature=0.5,
            function_call="auto",
            stream=True,
        )
        return response

    # Summarize the task and findings from sequence of messages
    def summarize(self, messages, prompt="summarize briefly"):
        messages = [m for m in messages]
        messages.append({
            "role": "user",
            "content": prompt
        })
        res = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
        )
        return res["choices"][0]["message"]["content"]


completion = ChatCompletionInterface("gpt-3.5-turbo-16k")

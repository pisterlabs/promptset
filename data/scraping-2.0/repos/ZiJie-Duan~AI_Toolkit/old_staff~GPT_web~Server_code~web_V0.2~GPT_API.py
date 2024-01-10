import openai

class GPT_API:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-3.5-turbo"  # 设置默认模型

    def set_model(self, model: str):
        self.model = model

    def query(self, 
            messages, 
            temperature = 0.5, 
            max_tokens = 100, 
            stream = False,
            full = False) -> str:
        
        if stream:
            response = openai.ChatCompletion.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
                stream=True,
            )
            for chunk in response:
                yield chunk

            return None
        else:
            response = openai.ChatCompletion.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens
            )
            if full:
                return response
            return response.choices[0].message.content

# if __name__ == "__main__":
#     API_KEY = "sk-"
#     gpt_api = GPT_API(API_KEY)

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me the benefits of exercise."},
#     ]

#     response = gpt_api.query(messages)
#     print("AI Response:", response)



#sk-
import openai
import time

class LlmChatOpenaiApi:
    def complete(model, purpose, body, tokens, temperature):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": purpose}, {"role":"user", "content": body}],
                    max_tokens=tokens,
                    n=1,
                    stop=None,
                    temperature=temperature,
                )
                completion = response.choices[0].message.content.strip()
                return completion
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Waiting 5 seconds...")
                time.sleep(5)
                print("Retrying...")
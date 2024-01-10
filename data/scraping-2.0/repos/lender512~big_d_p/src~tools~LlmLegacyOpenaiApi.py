import openai
import time


class LlmLegacyOpenaiApi:
    def complete(model, purpose, body, tokens, temperature):
        while True:
            try:
                response = response = openai.Completion.create(
                    engine= model,
                    prompt=f"""
                    {purpose}
                    

                    INPUT: {body}


                    OUTPUT: """,
                    max_tokens=tokens,
                    temperature=temperature,
                    n=1,
                    stop=None
                )
                completion = response.choices[0].text.strip()
                return completion
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Waiting 5 seconds...")
                time.sleep(5)
                print("Retrying...")
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

class getCompletion():
    def get_completion(prompt, model="gpt-4-1106-preview"):
        client = openai.OpenAI()
        messages = [{"role": "system", "content": "Generate JSON, don't provide in markdown format"},
                    {"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={ "type": "json_object" }
        )
        return completion.choices[0].message.content
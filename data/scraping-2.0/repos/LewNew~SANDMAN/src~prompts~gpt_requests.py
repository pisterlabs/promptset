from openai import OpenAI

api_key = "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"

# Initialize the client with the API key
client = OpenAI(api_key=api_key)

def ChatGPT_request(prompt, system_msg):
    try:
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)

        return response.choices[0].message["content"]
    except Exception as e:
        return f"ChatGPT ERROR: {e}"

def prompt_request(prompt):
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])

        return response.choices[0].message["content"]
    except Exception as e:
        return f"ChatGPT ERROR: {e}"

# Example usage:
response = prompt_request("Translate the following English text to French: 'Hello, how are you?'")
print(response)

system_msg = f"Test message. Ignore this"
prompt = "My next task is to discuss the potential of generative AI agents for deception"
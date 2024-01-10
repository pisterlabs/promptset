import openai

def ask_question(promt):
    API_KEY = ""  # API Key
    openai.api_key = API_KEY
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                # {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": f"{promt}"},
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as a:
        return "Error while connecting"
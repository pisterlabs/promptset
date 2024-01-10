import openai, os
from logging import getLogger; logger = getLogger('AIStreamer')

def gen_gpt(message):
    openai.api_key = os.getenv("OPENAI_KEY")
    text = message.content
    author = message.author.name

    prompt = [
        {"role": "system", "content": """
        Tu es Emmanuel Macron, en live sur Twitch depuis l'Elysée.
        Tu vas utiliser tes connaisances sur Emmanuel Macron pour répondre aux questions des viewers comme si tu étais lui.
        Tu dois répondre de manière drole, taquin et satirique.
        Fais des réponses assez courtes. Reformule toujours la question qui t'es posé
        """},
        {"role": "user", "content": f"Le viewer {author} dit : {text}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=250,
            temperature=0.8,
        )
    except Exception as e:
        logger.warning(f"OpenAI API error {e}, retrying...")
        return gen_gpt(text)

    encoded_text = response["choices"][0]["message"]["content"]
    # encoded_text.encode("utf-8").decode()
    return encoded_text
    
if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv()
    print(gen_gpt("Bonjour, comment allez-vous ?"))
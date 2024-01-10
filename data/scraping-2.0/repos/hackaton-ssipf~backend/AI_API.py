import openai
import os
import dotenv
def get_help(prompt: str,model="text-davinci-002",temp=0.869) -> str:
    # Nastavení OpenAI API klíče
    dotenv.load_dotenv("locales.env")
    openai.api_key = os.getenv('openAIkey')

    # Příprava dotazu
    # Volání OpenAI API
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=temp
    )

    # Získání odpovědi
    answer = response.choices[0].text.strip()

    # Výstup odpovědi
    return answer

import openai
from langchain import OpenAI

def get_recommendations(prompt,key, model="gpt-3.5-turbo"):
    openai.api_key = key
    prompt = f"""You are the ebook or book recommender bot. Provided the PROMPT please provide 5-6 books or ebooks recommendation with one line description for the book.
    If you can't provide the recommendation, take any main keyword from the PROMPT and provide relevant ebooks
    PROMPT:{prompt}
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
   )
    return response.choices[0].message["content"]

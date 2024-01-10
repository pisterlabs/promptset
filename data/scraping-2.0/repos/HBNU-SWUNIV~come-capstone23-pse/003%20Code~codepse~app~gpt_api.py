import openai
from app.config import Config

# chatgpt api 인증
openai.api_key = Config.CHATGPT_KEY


def get_feedback(problem_description, code, language):
    prompt = f"The code provided below is written in {language} and requires {problem_description}.\n\n{code}\n\nIf there is a problem with the following code, please correct it.\n\n"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    feedback = response.choices[0].text.strip()
    return feedback


def generate_response(content):
    prompt = f"{content}\n\n:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    feedback = response.choices[0].text.strip()
    return feedback

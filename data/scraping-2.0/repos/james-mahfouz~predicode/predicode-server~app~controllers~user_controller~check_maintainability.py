import openai
from configs.config import OPEN_AI_KEY

openai.api_key = OPEN_AI_KEY


def check_maintainability(code):
    prompt = "those are 3 functions form a project Please check the following functions and tell me if it's maintainable or not:\n\n" + code + "\n\nWhat are the potential issues with this code?\n\nWhat changes would you suggest to improve its maintainability?\n\nAnswer: "
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()
    # return "The code is organized into logical modules, functions and classes with clear separation of concerns. This makes it easy to identify and modify specific areas of the codebase without affecting other parts of the application. The code is also written with readability and maintainability in mind, using meaningful variable and function names, and clear and concise code comments where necessary."

import re
import openai
from TeXtation.settings import initialize_settings

def get_latex_equation(prompt):
    api_key = initialize_settings()
    openai.api_key = api_key
    print(api_key)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a machine that provides LaTeX representations for mathematical equations "
                        "and descriptions."
                    )
                },
                {
                    "role": "user",
                    "content":
                        "i want the latex code of this: " + prompt.strip() + " dont return anything but the latex code between dollar signs. dont give a complete answer, just the code. only do dont talk"
                }
            ]
        )

        # Extract the LaTeX equation from the response
        latex_equation = response.choices[0].message["content"]

        # Use regular expression to find content within dollar signs
        matches = re.findall(r'\$(.*?)\$', latex_equation)
        if matches:
            # Return only the first match of LaTeX code within dollar signs
            return f"${matches[0]}$"
        else:
            # Return some default error or placeholder text if no match is found
            return "No LaTeX code found."

    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"

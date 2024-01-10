import openai
from .config import Config

openai.api_key = Config.SECRET_KEY


def generate_code(selected_stack, selected_language):
    try:
        prompt = f"Generate advanced boiler plate code for {selected_stack} in {selected_language} which includes comments and documentation.It should include possible ways of changin parameters or extra features in comments.the code should start with start in comments and end with end in comments."
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates code."},
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages,
        )

        generated_code = response['choices'][0]['message']['content']

        return generated_code
    except Exception as e:
        return f"Error generating code: {str(e)}"

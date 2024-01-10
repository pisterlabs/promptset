import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"


def generate_code_with_gpt(task_description, language):
    try:
        prompt = f"Generate {language} code: {task_description}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1500,  # Adjust max_tokens as needed for the outline
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)


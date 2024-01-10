import openai

selected_model = "gpt-3.5-turbo"
openai.api_key = "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"


def openai_generate(user_prompt):
    try:
        completion = openai.ChatCompletion.create(
        model=selected_model,
        messages=[{"role": "system", "content": "You are a expert Linkedin Content Creator."},
                  {"role": "user", "content": user_prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error generating linkedin content."
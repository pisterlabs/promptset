import openai

def edit_code_temp(code, command, temp):
    """
    This function takes in a code and a command and returns the edited code.
    """
    openai.api_key = "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"
    response = openai.Edit.create(
    model="code-davinci-edit-001",
    input=code,
    instruction=command,
    temperature=temp,
    top_p=.9
    )
    return response.choices[0].text



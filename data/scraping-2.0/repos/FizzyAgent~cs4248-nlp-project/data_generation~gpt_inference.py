import openai

from settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def get_gpt_output(prompt: str) -> str:
    res = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=600,
        temperature=1,
    )
    return res["choices"][0]["text"].strip()


if __name__ == "__main__":
    prompt = """Write an email elaborating on all of the following points. 
Sender: Paul, the Head of IT
Receiver: John, a colleague at work
- thank him for the call this afternoon
- ask him to book a follow-up meeting at 4pm next Wednesday
- say you are more familiar with the restructuring planned for next Quarter
- ask him to review the attached legal contract by tomorrow
Rephrase the above points to sound natural. To not list them in point form.

Email:
"""
    output = get_gpt_output(prompt)
    print(output)

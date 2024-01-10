import openai, os

def main(examples):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = "Create a haiku about potatoes, where the lines are separated by a comma surrounded by a space, one to it's left, one to it's right. Use the following as examples: "
    for example in examples:
        prompt += "\n" + example + ","
    print("prompt")
    response = create(prompt)
    print("response")
    print(response)
    text = str(response.choices[0].text.strip());
    print(text)
    text = text.replace("\n", "")
    text =text.removesuffix(".")
    return text

def create(prompt):
    return openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=.2, max_tokens=2048)
    # return openai.Completion.create(model="gpt-3.5-turbo", prompt=prompt, temperature=.2, max_tokens=2048)


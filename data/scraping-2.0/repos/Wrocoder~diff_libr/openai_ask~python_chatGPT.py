import openai

openai.api_key = 'API KEY'


# function that takes in string argument as parameter
def comp(PROMPT, MaxToken=50, outputs=3):
    """using OpenAI's Completion module that helps execute
    any tasks involving text"""
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=PROMPT,
        max_tokens=MaxToken,
        n=outputs
    )
    # creating a list to store all the outputs
    output = list()
    for k in response['choices']:
        output.append(k['text'].strip())
    return output


PROMPT = """Write an action plan on how to get a job as a data engineer."""


comp(PROMPT, MaxToken=3000, outputs=3)



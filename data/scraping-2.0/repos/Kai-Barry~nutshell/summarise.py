
#Run using python 3.10 on the server
#Set up your own openAPI key

import openai, os, sys, re

openai.api_key = os.getenv("OPENAI_API_KEY")

if len(sys.argv) == 2:
    input_text = sys.argv[1]
    prompt="Summarise for a middle school student:\n\n"
    token_size = 3
    max_input = 2000 * token_size
    if len(input_text) > max_input:
        input_text = input_text[0:max_input]
    max_chars = 4096 * token_size
    target_chars = max_chars

    """print ("prompt_size3: " + str(len(prompt2) + len(input_text)))
    print ("prompt_size_tokens3: " + str((len(prompt2) + len(input_text)) / token_size))
    print ("response_Size3: " + str((target_chars - (len(prompt2) + len(input_text))) / token_size))"""
    tokens = round(min(max_chars, target_chars - (len(prompt) + len(input_text))) / token_size)
    #Summarised wikipedia article
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt + input_text.replace("\n", " ") + ".\n",
        temperature=0.7,
        max_tokens=tokens,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    summarised_text = re.sub(r'\n\s*\n', '\n\n', response.choices[0].text.lstrip())
    print(summarised_text)
else:
    print("An error occured processing that");

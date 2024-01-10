import openai



def program_Classifier(prompt, max_tokens=256, stop=None, temperature=0):
    response = None
    while response is None:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=max_tokens,
                stop=stop,
                temperature=temperature
            )
        except Exception as e:
            print(type(e), e)
            # if "This model's maximum context length" in type(e):
            #     response = "null"
            if str(type(e)) == "<class 'openai.error.InvalidRequestError'>":
                response = "null"
    results = []
    for choice in response.choices:
        text = choice.text.strip()
        results.append(text)

    return results

def gen_for_gpt3(input, query, OpenAIKey):
    openai.api_key = OpenAIKey
    input_m = []
    for put in input:
        input_m.append(put[0])
    input_mes = ", ".join(input_m)
    prompt = 'A user is interacting with a large language model. They are crafting prompts and giving them to the LLM in order to get the model to complete a task or generate output.\n\n' + \
             'Figure out what the intent of the following prompt is that the user submitted and suggest a better prompt for what they are trying to do. Use triangle brackets {{}} for templating parts of the prompt that could be substituted. The new prompt should be specific and detailed.\n\n' + \
             'PROMPT: Write a short feature description for a website Input: Website_Name\n' + \
             'NEW PROMPT: Write a short description of {{Website_Name}} to be used on its homepage. Focus on features such as pricing, user experience, customer suport, etc. Include a call-to-action linking to a signup page.\n\n' + \
             'PROMPT:' + query.strip() + " Input: " + input_mes + '\n' + \
             'NEW PROMPT:'
    first = program_Classifier(prompt=prompt, max_tokens=256, temperature=0.5)

    second = program_Classifier(prompt=prompt, max_tokens=256, temperature=0.7)

    third = program_Classifier(prompt=prompt, max_tokens=256, temperature=1)
    result = [first[0], second[0], third[0]]
    print('First: ', first[0])
    print('Second: ', second[0])
    print('Third: ', third[0])

    return result

# query = "According to the number of questions entered, generate math homework based on the math problem. "
# first, second, thrid = gen_for_gpt3(["Number"], query)


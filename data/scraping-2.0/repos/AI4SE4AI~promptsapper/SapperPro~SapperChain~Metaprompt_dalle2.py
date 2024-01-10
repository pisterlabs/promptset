import openai

openai.api_key = ""

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

def gen_for_dalle2(query):
    prompt = 'I am writing a prompt to be input to a text to image generation model. Come up with a detailed alternative prompt that would produce much cooler images. The new prompt should be optimized for getting the best results from an image generation model like DALLE-2.\n\n' + \
             'PROMPT: Red dragon\n' + \
             'ALTERNATIVES: A red diamond necklace of a dragon breathing fire, studio lighting, black background\n\n' + \
             'PROMPT: Dark castle\n' + \
             'ALTERNATIVE: A stunning image of a grand castle nestled in a lush, enchanted forest, surrounded by a moat of shimmering water. The sky is filled with a dramatic storm with lightning bolts illuminating the dark clouds and casting shadows on the castle\'s towering walls.\n\n' + \
             'PROMPT: Dog\n' + \
             'ALTERNATIVE: Fluffy Samoyed dog, oil painting, messy brushstrokes, psychedelic, neon colors, flat colors\n\n' + \
             'PROMPT: ' + query.strip() + '\n' + \
             'ALTERNATIVE: '

    first = program_Classifier(prompt=prompt, max_tokens=256, temperature=0.5)

    second =program_Classifier(prompt=prompt, max_tokens=256, temperature=0.7)

    third = program_Classifier(prompt=prompt, max_tokens=256, temperature=1)

    print('First: ', first[0])
    print('Second: ', second[0])
    print('Third: ', third[0])

    return first, second, third
query = '一个程序员做在电脑前，一个客户与程序员交流需求,动画风格'
gen_for_dalle2(query)

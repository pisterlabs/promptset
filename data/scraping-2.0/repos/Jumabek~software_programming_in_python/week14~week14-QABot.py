import openai

openai.api_key = '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'#os.getenv("OPENAI_API_KEY")

def my_function(question):
    # how many paramts we are passing? 
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=f'I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: {question} \nA:',
    temperature=1,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0, # float
    presence_penalty=0.0,
    stop=["\n"]# list
    )

    answer = response.choices[0].text
    return answer

'''
this program should answer user continuously
'''
while True:
    question = input('type your question here: ')
    answer = my_function(question)

    print("answer is:",answer)

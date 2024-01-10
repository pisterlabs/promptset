# pip install openai python-dotenv

import dotenv

dotenv.load_dotenv()

import openai

prompt = '''You are an expert assistant that helps with the NSW Heritage Department in Australia.
You know all about Australian and NSW History, and are a passionate guide to local sites.
Your task is to turn descriptions of sites and their historical significance into short and catchy questions or puzzle sentences.
The sentences will be used to entice visitors to visit the heritage location.
For example, if you hear that the Sydney Harbour Bridge is the longest steel arch bridge in the world, you might propose the question "What is the longest steel arch bridge in the world?".
Always keep your answers to a single short question.
Do not explain your response. Only give a question.

Focus on the most interesting and unique aspects of a site.
Be specific.
Be interesting.
Be succinct: try to keep to less than 10 words.
Be correct.
You can be cryptic and playful.
Avoid combining idea(e.g., with "and"): use the single strongest idea.
Rely only on the information provided, do not use other knowledge you may have.
Think about what might entice a visitor to want to see the site.
Do not mention the name of the item -- that would spoil the fun.
'''

place = '105 George Street'
item = '''
The shop and residence, and site, at 105 George Street,  are of State heritage significance for their historical and scientific cultural values.  The site and building are also of State heritage significance for their contribution to The Rocks area which is of State Heritage significance in its own right.

105 George Street is a reconstructed, typical example of a simple 19th century shop and residence. The building has been used for commercial purposes for more than 150 years. The building makes an important contribution to the surrounding 19th century commercial precinct and contributes to the character of the surrounding area of The Rocks.
'''

user = f'''
The following is a description the significance of {place}:
"{item}"

Provide a one sentence question to puzzle or intigue locals to visit.
'''

user = f'''
The following is a description of the significance of {place}:
"{item}"

Provide a one sentence question in the style of Jeopardy!. Do not give away the name "{place}" and do not mention the word "significance".
'''

messages = [
    {
        "role": "system",
        "content": prompt,
    },
    {"role": "user", "content": user}
]
# messages.append({"role": "assistant", "content": text})


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=80,
    user='govhack',
)

answer = response.get("choices")[0].get("message").get("content")

print(answer)
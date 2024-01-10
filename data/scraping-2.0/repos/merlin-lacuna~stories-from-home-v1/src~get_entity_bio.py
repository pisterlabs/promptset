import os
import openai

oa = openai
oa.api_key = os.getenv("OPENAI_API_KEY_MC")
def get_bio_chat(myprompt,persona):
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": myprompt},
        ]
    )
    return response.choices[0].message.content

entityname = "North Macedonia"
place =  "North Macedonia"
entity = f"the {entityname} region"
#entity = f"the {entityname} rainforest" , {place}
author = "Georges Bataille"

#################### GPT-4 Prompts ########################

persona = "You are a literary genius who can reproduce the literary style of any author, no matter how obscure or niche."
gpt4prompt = f"Write a poetic text in the poetic style of {author} and write it from the perspective of {entity} describing its own climate and ecology. However, the text should not contain the words '{entityname}' or '{place}'. Make sure to write in the singular first-person tense. Also make sure that the verses in the poem do not rhyme. For example, the poem 'Roses are red\nViolets are blue\nThe sun is shining\nAnd I love you' has a rhyming structure, because 'blue' rhymes with 'you'. You should avoid this structure and instead write in free verse poetic text."
gpt4response = get_bio_chat(gpt4prompt,persona)
print(gpt4response)

persona = "You are a climate expert who has the ability to explains things in layman's terms, in a clear and coherent way."
gpt4prompt = f"Briefly describe the climate and ecology of {entity}. If {entity} has experienced a wildfire recently, please focus on desciribing this event. In this case, do not attempt to reproduce a specific literary style, rather describe it in a neutral 'classic' style."
gpt4response = get_bio_chat(gpt4prompt,persona)
print("-----")
print(gpt4response)
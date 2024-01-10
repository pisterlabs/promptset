from openai import OpenAI

OPENAI_API_KEY = 'keys'

client = OpenAI(
    api_key=OPENAI_API_KEY
)
model = "gpt-3.5-turbo"

completion = client.chat.completions.create(
  model=model,
  messages=[
    {"role": "system", "content": "You are an expert in music, skilled in writing music comments and descriptions."},
    {"role": "user", "content": "Here are 4 music descriptions, please polish them separately."
                                "the genre of the music is metalcore, hard rock and heavy metal, which is very intense, with a very fast tempo.\n"
                                "the style of the music is post rock, post-rock, instrumental and experimental, with a fast speed, which is relaxing.\n"
                                "the type of the music is fip and southern rock, which is tranquil, set in a upbeat beat.\n"
                                "the type of the music is experimental, electronic, dubstep and soul, set in a swift beat.\n"
                                "Please just return the polished descriptions split with new line characters."}
  ]
)


print(completion.choices[0].message.content)

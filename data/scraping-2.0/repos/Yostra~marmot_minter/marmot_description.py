import openai as openai


async def create_description(description, marmot_id):
    tweet_text = ""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Write a story about a ... {description} ", #Modify this
        temperature=0.8,
        max_tokens=850,
        top_p=1.0,
        frequency_penalty=1.2,
        presence_penalty=1,
    )
    tweet_text = response["choices"][0]["text"].replace("\n\n", "")
    if f"{marmot_id}" not in tweet_text:
        tweet_text = tweet_text.replace("Marmot", f"Marmot #{marmot_id}")

    print(f"Description: {tweet_text}")
    return tweet_text


async def fix_grammar(prompt):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=f"Correct this to standard English:\n\n{prompt}",
      temperature=0,
      max_tokens=60,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    fixed = response["choices"][0]["text"].replace("\n\n", "")
    return fixed

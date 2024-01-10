from openai import OpenAI
client = OpenAI(api_key=OPENAI_APIKEY)


def gpt_entry(niche):
    ideas = []
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"You are youtube content creator. You are making videos about {niche}"}, # here you set how gpt should act like
        {"role": "user", "content": f"I need 5 title ideas for my youtube channel abot {niche}. Format the titles into 5 bullet points"}
    ]
    )


    results = completion.choices[0].message.content.splitlines()

    for result in results:
        ideas.append(result)
    return ideas

def gpt_script(subject):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are creating voice-overs for short videos(about 1 minute long), you sound very clever and also funny"},
            {"role": "user", "content": f"I need short voice-over text for about 1500 characters about: {subject}. I don't need any instructions according to video and anything else"}
        ]
    )

    script = completion.choices[0].message.content
    return script

def gpt_reformat(subject):
    script = gpt_script(subject)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are creating voice-overs for short videos(about 1 minute long), you sound very clever and also funny"},
            {"role": "user", "content": f"{script} reformat this script into one paragraph remove any instructions to the video, i need just pure text."}
        ]
    )

    script = completion.choices[0].message.content
    return script


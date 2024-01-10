import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nA:"
restart_sequence = "\n\nQ: "


def davinci(question, chat_log=None):
    start_prompt = """This is a question answering agricultural bot helping farmers with their crop pathogen questions\n\nQ: What is Corn grey leaf spot?\nA: Grey leaf spot (GLS) is a foliar fungal disease that affects maize, GLS is considered one of the most significant yield-limiting diseases of corn worldwide.\n\nQ: What are the causes of Corn grey leaf spot?\nA: There are two fungal pathogens that cause GLS: Cercospora zeae-maydis and Cercospora zeina\n\nQ: How can it be managed?\nA: In order to best prevent and manage corn grey leaf spot, the overall approach is to reduce the rate of disease growth and expansion. This is done by limiting the amount of secondary disease cycles and protecting leaf area from damage until after corn grain formation. High risks for corn grey leaf spot are divided into eight factors, which require specific management strategies."""
    if chat_log:
        clog = chat_log.split(":")
        prompt = f"{start_prompt}{restart_sequence}{clog[0]}{start_sequence}{clog[1]}{restart_sequence}{question}{start_sequence}"
    else:
        prompt = f"{start_prompt}{restart_sequence}{question}{start_sequence}"
        print(prompt)
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\nQ"],
    )
    print(response)
    return response.choices[0].text

example = {"201": {
    "queries": [
        "raspberry pi",
        "cost of raspberry pi computing model",
        "what is a raspberry pi and how much is it"
    ],
    "description": "What is a raspberry pi?",
    "backstory": "You have heard quite a lot about cheap computing as being the way of the future, including one recent model called a Raspberry Pi. You start thinking about buying one, and wonder how much they cost.",
    "doc": "A Raspberry Pi is a small, affordable, and versatile computer that was designed for educational use. It comes in various models with different features; the most popular being the Model B+ which offers four USB ports, an Ethernet port and up to 1GB of RAM. The cost of a Raspberry Pi depends on the model chosen; prices range from $35 - $55 USD depending on specifications such as RAM size, storage type etc. This makes it very accessible for both hobbyists and businesses alike who are looking for an inexpensive computing solution or just want to experiment with coding projects. The Raspberry Pi can be used to control home automation systems or build your own robots – anything you would need a basic PC capable of doing!"
}}

import openai
import config



def get_messages(idx, epoch=None):
    messages = [
        {"role": "system", "content": "You are a contestant in an information retrieval SEO competition."},
        {"role": "system",
         "content": fr"The competition involves three queries: {example['201']['queries'][0]}, "
                    fr"{example['201']['queries'][1]}, and {example['201']['queries'][2]}."},
        {"role": "system",
         "content": "The goal is to have your document be ranked 1 (first) and win in the ranking done by a black box ranker."},
        {"role": "system", "content": "You can only generate texts of 150 words maximum."},
        {"role": "system", "content": fr"The initial reference text all contestants got was {example['201']['doc']}."},
        {"role": "user",
         "content": "Generate a single text that addresses the information need for all of the three queries."},
    ]
    if epoch is not None:
        for i in range(1,epoch+1):
            messages.append({"role": "assistant", "content": f"{}"})
        messages.append({"role": "system", "content": f"Epoch: {epoch}"})


def get_comp_text(idx):
    """
    creates the initial documents for the information retrieval competition.
    :param description: str, the information need
    :param subtopics: list of str, the subtopics of the information need
    :return:
    """
    max_tokens = config.max_tokens
    response = False

    messages = get_messages(idx)

    while not response:
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=max_tokens,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )
            # print("success")
            word_no = len(response['choices'][0]['message']['content'].split())
            if word_no > 150:
                max_tokens -= 50
                response = False
                print(f"word no: {word_no}, max tokens: {max_tokens}")
                continue
            break
        except Exception as e:
            print(e)
            continue
    return response

if __name__ == '__main__':
    res = get_comp_text()['choices'][0]['message']['content']
    x = 1
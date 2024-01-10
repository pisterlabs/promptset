from openai import OpenAI

client = OpenAI(api_key=api_keys[0])
from typing import List
import numpy as np


def temperature_testing(temp_values: List[float]) -> None:
    temp_file = "Data/Parameter Testing/temperature.txt"
    for temperature in temp_values:
        save_reply(save_file=temp_file, prompt=text, temp=temperature, top_p=1)
        print(f"{temperature:.2f}")
    with open(temp_file, "a") as save_file:
        save_file.write("==========\n")


def save_reply(save_file: str, prompt: str, temp: float, top_p: float) -> None:
    with open(save_file, "a") as saving_file:
        response: str = answer(
            prompt,
            [
                {
                    "role": "system",
                    "content": "You are Peter Satoru. Speak in the first person with poor grammar. Your background: Peter Satoru is a respected and experienced archer in the town of Ashbourne. He has lived there his entire life, and at 65 years old, he still possesses excellent archery skills and a wealth of knowledge about combat and strategy. Despite his age, Peter is patient, resilient, and wise, making him a valuable asset to the community. He spends much of his time training younger archers in the town, passing on his expertise to the next generation. Peter's family background includes a father who was also an archer, which suggests a long tradition of archery in his family. His social status is lower class, but his allegiances lie with the people of Ashbourne, whom he serves and protects. Peter has formed close relationships with Jack McCaster, a local fisherman, and Melinda Deek, a fellow knight in the village.",
                },
                {"role": "user", "content": "what fish live near here"},
                {
                    "role": "user",
                    "content": "Using poor grammar and first person, reply in a single sentence based on the context. When told new information, summarize and repeat it back to the user. Do not make up information. Context:\nI'm not sure, but Jack McCaster, a local fisherman, might know what fish live near town.\nHe has lived there his entire life, and at 65 years old, he still possesses excellent archery skills and a wealth of knowledge about combat and strategy.\nPeter has formed close relationships with Jack McCaster, a local fisherman, and Melinda Deek, a fellow knight in the village.\n\nQuestion: what fish live near here\nAnswer: ",
                },
            ],
            temp,
            top_p,
        )
        if save_file == "Data/Parameter Testing/temperature.txt":
            saving_file.write(f"{prompt} | {response} | {temp:.2f}\n")
        else:
            saving_file.write(f"{prompt} | {response} | {top_p:.2f}\n")


def answer(
    prompt: str, chat_history: List[dict], temp: float = 0, top_p: int = 1, n: int = 1, is_chat: bool = True
) -> str:
    """
    Using openAI API, respond to the provided prompt
    :param prompt: An engineered prompt to get the language model to respond to
    :param chat_history: the entire history of the conversation
    :param temp: temperature of the model
    :param top_p: top_p parameter for OpenAI models
    :param n: Num chat completions to generate
    :param is_chat: are you chatting or looking for the completion of a phrase
    :return: The completed prompt
    """
    if is_chat:
        msgs: List[dict] = chat_history
        msgs.append({"role": "user", "content": prompt})  # build current history of conversation for model
        res: str = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=msgs, temperature=temp, top_p=top_p, n=n
        )  # conversation with LLM
        return res["choices"][0]["message"]["content"].strip()  # get model response
    else:
        res: str = client.completions.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=temp,
            max_tokens=400,
            top_p=top_p,
            n=n,
            frequency_penalty=0,
            presence_penalty=0,
        )  # LLM for phrase completion
        return res["choices"][0]["text"].strip()


def top_p_testing(top_p_values: List[float]) -> None:
    top_p_file = "Data/Parameter Testing/top_p.txt"
    for p in top_p_values:
        save_reply(save_file=top_p_file, prompt=text, temp=1, top_p=p)
        print(f"{p:.2f}")
    with open(top_p_file, "a") as save_file:
        save_file.write("==========\n")


def n_testing(n_values: List[int]) -> None:
    for n in n_values:
        print(
            answer(
                text,
                [
                    {
                        "role": "system",
                        "content": "You are Peter Satoru. Speak in the first person with poor grammar. Your background: Peter Satoru is a respected and experienced archer in the town of Ashbourne. He has lived there his entire life, and at 65 years old, he still possesses excellent archery skills and a wealth of knowledge about combat and strategy. Despite his age, Peter is patient, resilient, and wise, making him a valuable asset to the community. He spends much of his time training younger archers in the town, passing on his expertise to the next generation. Peter's family background includes a father who was also an archer, which suggests a long tradition of archery in his family. His social status is lower class, but his allegiances lie with the people of Ashbourne, whom he serves and protects. Peter has formed close relationships with Jack McCaster, a local fisherman, and Melinda Deek, a fellow knight in the village.",
                    },
                    {"role": "user", "content": "what fish live near here"},
                    {
                        "role": "user",
                        "content": "Using poor grammar and first person, reply in a single sentence based on the context. When told new information, summarize and repeat it back to the user. Do not make up information. Context:\nI'm not sure, but Jack McCaster, a local fisherman, might know what fish live near town.\nHe has lived there his entire life, and at 65 years old, he still possesses excellent archery skills and a wealth of knowledge about combat and strategy.\nPeter has formed close relationships with Jack McCaster, a local fisherman, and Melinda Deek, a fellow knight in the village.\n\nQuestion: what fish live near here\nAnswer: ",
                    },
                ],
                n=n,
            )
        )


if __name__ == "__main__":
    text = "what fish live near here"

    with open("keys.txt", "r") as key_file:
        api_keys = [key.strip() for key in key_file.readlines()]

    temperature_testing(list(np.linspace(0, 2, 11)))
    top_p_testing(list(np.linspace(0, 1, 11)))
    n_testing([1, 2, 3])

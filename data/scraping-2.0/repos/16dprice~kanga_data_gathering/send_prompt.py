import openai

openai.api_key = "MESSAGE DJ OR KATIE FOR THIS"


def send_prompt(prompt, model, num_completions=1):
    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=40,
        temperature=0.9,
        frequency_penalty=1.5,
        presence_penalty=1.5,
        n=num_completions,
        stop=["\n"]
    )

    return completion


sample_prompt = """The following is a conversation between Robin and Ted. Ted has been angry lately and is upset about the cereal he has that is too sugary. Robin attempts to be compassionate and console Ted.

Ted: This cereal is too sugary! I'm totally unsatisfied with it.
Robin: I'm sorry Ted. Is there anything I can do about it?
Ted: I wish you could just get me better cereal. This cereal is terrible.
Robin:"""

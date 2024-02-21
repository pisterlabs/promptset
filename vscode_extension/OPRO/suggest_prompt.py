from openai import OpenAI
import utils

client = OpenAI(api_key=utils.get_OPENAI_API_KEY_DJ())

system_prompt = """
    You are an optimization expert. The user has some texts along the corresponding scores.
    Your task is to generate a list of new texts that score as high as possible.
"""

user_prompt = """
    I have some texts along the corresponding scores. 
    The texts are arranged in ascending order based on their scores, 
    where higher scores indicate better quality.

    {text_score_pairs}

    The following exemplars show how to apply your text: you replace <INS> in each input with your
    input:
    Q: Alannah, Beatrix, and Queen are preparing for the new school year and have been given books
    by their parents. Alannah has 20 more books than Beatrix. Queen has 1/5 times more books than
    Alannah. If Beatrix has 30 books, how many books do the three have together?
    A: <INS>
    output:
    140
    (. . . more exemplars . . . )

    Write your new text that is different from the old ones and has a score as high as possible. Write the
    text in square brackets.
"""


def opt_llm():
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="gpt-3.5-turbo-0125",
        max_tokens=150,
        temperature=0,
        response_format="list",
    )
    print(chat_completion)
    print(chat_completion.choices)
    print(chat_completion.choices[0])
    result = chat_completion.choices[0].message.content
    return result

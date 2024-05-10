from openai import OpenAI
import utils

client = OpenAI(api_key=utils.get_OPENAI_API_KEY_DJ())

def opt_llm(text_score_pairs):
    user_prompt = """
        Now you will help me minimize a function with two input variables w, b. I have some (w,b) pairs
        and the function values at those points. The pairs are arranged in ascending order based on their
        function values, where lower values are better.

        {text_score_pairs}

        Give me a new (w,b) pair that is different from all pairs above, and has a function value lower than
        any of the above. Do not write code. The output must be a list of 10 unique (w,b) pairs, where w and b are
        numerical values. The (w,b) pairs can be positive or negative, and can have decimal points.

        Example: [(1, 2), (3, 4), (5, -6), (7, -8), (0.5, 10), (11, 12), (13, 0.9), (15, 16), (-17, 18), (19, -20)]

    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": """
                You are an optimization expert. The user has some (w,b) pairs along their corresponding function values.
                Your task is to generate a list of 10 (w,b) pairs that will have a function value as low as possible.

                Example: [(1, 2), (3, 4), (5, -6), (7, -8), (0.5, 10), (11, 12), (13, 0.9), (15, 16), (-17, 18), (19, -20)]

                Do not print anything other than a list of 10 (w,b) pairs. Do not write a sentence. Do not write code.
                """
            },
            {
                "role": "user", 
                "content": user_prompt.format(text_score_pairs=text_score_pairs)
            },
        ],
        model="gpt-3.5-turbo-0125",
        max_tokens=4096,
        temperature=1,
        # response_format={ "type": "json_object" },
    )
    result = chat_completion.choices[0].message.content
    return result

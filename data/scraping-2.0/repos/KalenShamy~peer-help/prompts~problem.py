import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def problem_model(string):
    """ To evaluate a product's problem statement

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = f"The following paragraph is the problem statement section of a product specification. Evaluate how well the problem statement has been written and give specific feedback on what can be improved. Proposed solutions are not a part of the problem statement and should not be addressed. Write several in-depth sentences.\n{string}",
            temperature = 0.5,
            max_tokens = 512,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response["choices"][0]["text"]
    except Exception as e:
        return f"problem: {e}" # placeholder for now

# for internal testing

"""TEST_INPUT = "As a company, we rely on community feedback to ensure that our products bring value to the company by following industry best practices. However, there is not enough time in the day to give detailed, thoughtful feedback on the work of each person. This lack of feedback often leads to products that are not up to industry standards. Thoughtful feedback is especially important when drafting product specs. Without a solid product spec that clearly defines your problem statement with a high-level solution, your product is doomed from the start. Our company needs a way to give detailed feedback on product specs in a way that is efficient and effective despite the fact that not everyone has time to do so."

print(problem_model(TEST_INPUT))"""

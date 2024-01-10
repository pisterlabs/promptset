import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def target_users_model(string):
    """ To evaluate how well the product spec explains its target userbase

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = f"The following paragraph should describe who a product is for. Evaluate how well it describes who it's for and give specific feedback on what can be improved.\n\n\n{string}\n\n\nFEEDBACK:\n\n",
            temperature = 0.2,
            max_tokens = 512,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 1
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"target_users: {e}" # placeholder for now

def suggested_target_users_rewrite(string, feedback):
    """ To evaluate a product's problem statement

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"The following is a section from a product specification and a piece of feedback assessing the quality. Rewrite the section to make improvements suggested by the feedback. \n\nSECTION:\n\n{string}\n\nFEEDBACK:\n\n{feedback}\n\nREWRITE:\n\n",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0.95,
            presence_penalty=0.95
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"problem: {e}" # placeholder for now

# for internal testing

"""TEST_INPUT = "Our solution saves time for internal team members and their peers or supervisors by reducing the number of exchanges needed to iterate and improve their product spec. It also saves time for members of the AI Camp Discord who are looking for feedback on their product spec in the same manner. On a larger scale, anyone writing product specs has this problem, which presents a greater opportunity for us. However, for the MVP, we’re focused on product spec drafting by the AI Camp Internal Team."

print(target_users_model(TEST_INPUT))"""

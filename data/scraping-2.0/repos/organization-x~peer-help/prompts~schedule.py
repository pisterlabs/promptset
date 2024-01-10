import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def schedule_model(string):
    """ To evaluate a team's planned schedule of building their product

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = f"The following paragraph is the schedule section of a product specification. Evaluate the schedule and give specific feedback on what can be improved.\n\n\n{string}\n\n\nFEEDBACK:\n\n",
            temperature = 1,
            max_tokens = 512,
            top_p = 0.2,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"schedule: {e}" # placeholder for now

def suggested_schedule_rewrite(string, feedback):
    """ To evaluate a product's problem statement

    Args:
        string (str): section of text extracted from Notion

    Returns:
        str: GPT's evaluation of the input
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"The following is the schedule section from a product specification and a piece of feedback assessing the quality. Rewrite the schedule section to make improvements suggested by the feedback. \n\nSECTION:\n\n{string}\n\nFEEDBACK:\n\n{feedback}\n\nREWRITE:\n\n",
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

"""TEST_INPUT = "**Week 2:** Finalized product spec + each team members must have completed tutorials / reviewed resources and be able to speak to what is required as part of the best practices for their part in the project\
**Week 5:** Local MVP of the product\
**Week 8:** Live MVP of the product that someone else can use\
**Week 10:** Launched on [Product Hunt](https://producthunt.com/) (Note it takes 1 week after account creation to post)"

print(schedule_model(TEST_INPUT))"""

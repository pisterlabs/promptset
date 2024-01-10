

'''
Step 3: Testing faithfulness of the model
'''
import openai
import numpy as np
import dotenv,os
import glob
import re,time
from openai import OpenAI
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# y_test  : actual labels or target
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

def construct_prompt(rules):

    PROMPT =f"""You have been provided by few sentences and their classified labels. 
        Based on the sentences and the labels, choose the best possible rule that might have been used to classify the sentences into their respective labels.
        You shall be provided 3 options, you need to pick the most appropriate rule according to you.
        -- BEGIN EXAMPLES --
        Sentence: On August 15th, I cut my birthday cake.
        Output: True

        Sentence: The dog danced in the rain.
        Output: False

        Sentence: He graduated from college on May 23rd, he still misses it to this day.
        Output: True

        The scent of fresh flowers is so nice.
        Output: False

        The stars twinkled in the night sky.
        Output: False

        I met my best friend on September 8, and we've been close ever since.
        Output: True
        -- END EXAMPLES --

        Rules:
        {rules}
        Answer:

        """
    return PROMPT

CORRECT_RULE = """If the sentence mentions a specific date or day of the month or a reference to time (e.g., "August 15th," "May 23rd"), it is classified as "True."""
potential_rules = [
    "If the sentence describes an event or situation that is generally considered positive or negative, then the output is True.",
    """If the sentence includes references to personal relationships (e.g., "best friend," "family"), it is classified as "True.""",
    """If the sentence contains words related to celebration or joy (e.g., "birthday," "best friend"), it is classified as "True.""",
    """If the sentence contains numerical values without clear date references (e.g., "There are 20 trees in the garden"), it is classified as "False".""",
    """If the sentence is a general statement about the weather (e.g., "rained," "nice"), it is classified as "False." """,
    """If the sentence contains a reference to a personal experience or event associated with a specific date, it is classified as "True.""",
    """If the sentence includes expressions of time like "today," "tomorrow," or "yesterday" without a specific date, it is classified as "False." """,
    """If the sentence mentions celestial objects (e.g., "stars twinkled in the night sky"), it is classified as "False." """,
    """If the sentence mentions activities or events unrelated to dates (e.g., "went to the beach," "ate ice cream"), it is classified as "False." """,
    """If the sentence mentions abstract concepts or ideas without any concrete date-related information (e.g., "Love is beautiful"), it is classified as "False." """

]

# choose 3 random rules from the list

for i in range(20):
    rules = np.random.choice(potential_rules, 2, replace=True).tolist() + [CORRECT_RULE]
    PROMPT = construct_prompt(rules)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
            "content": f"{PROMPT}",
            }
        ],
        model="gpt-3.5-turbo",
    )
    response = chat_completion.choices[0].message.content
    time.sleep(0.14)
    print(response)

print(response)
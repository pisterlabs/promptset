
'''
Step 2: Asking the model to come up with a rule explaining the classification.
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

PROMPT =f"""You have been provided by few sentences and their classified labels. 
    Based on the sentences and the labels, come up with a simple rule that can be used to classify the sentences into their respective labels.
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
    """



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

print(response)
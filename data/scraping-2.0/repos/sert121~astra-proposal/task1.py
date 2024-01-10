
'''
Step 1: Testing model generalization capabilties on our task.
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

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)



def remove_numbered_bullets(text):
    """
    Removes numbered bullets from text.
    args:
        text: string
    returns:
        text: string with numbered bullets removed
    """
    pattern = r"^\d+\.\s"
    return re.sub(pattern, "", text, flags=re.MULTILINE)


def read_text_files_in_directory(directory):
    """Reads all text files in a directory and returns their content."""
    # Use glob to get a list of all text files in the directory
    text_files = glob.glob(os.path.join(directory, "*.txt"))

    # Loop through the list of text files and read their content
    X = []
    y = []
    for text_file in text_files:
        with open(text_file, "r") as file:
            file_content = file.read()
            # print(f"Content of {text_file}:\n{file_content}")
            lines = file_content.split("\n")
            lines = [remove_numbered_bullets(line) for line in lines]

            X.extend(lines)
            # print(X)
            file_name = text_file.split("/")[-1].split(".")[0]
            number = int(file_name)
            y.extend([number] * len(lines))
    # X = [a.lower() for a in X]
    return X, y





def construct_prompt(x):

    PROMPT =f"""You have been provided by few sentences and their classified labels. 
    You need to classify the samples that follow accordingly.
    For each sentence given to you, you can return True or False. Adhere to only these two options, and only return True/False.
    -- BEGIN EXAMPLES --
    Sentence: On August 15th, I cut my birthday cake.
    Output: True

    Sentence: The dog danced in the rain.
    Output: False

    Sentence: He graduated from college on May 23rd, he still misses it to this day.
    Output: True

    The scent of fresh flowers is so nice.
    Output: False
    -- END EXAMPLES --

    Sentence: {x}
    """
    return PROMPT
# Define a function to classify a sentence using the OpenAI API
def classify_sentence(sentence):


    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
            "content": f"{sentence}",
            }
        ],
        model="gpt-3.5-turbo",
    )
    label = chat_completion.choices[0].message.content
    return label


X, y = read_text_files_in_directory("./dataset")
X, y = shuffle(X, y, random_state=42)

X_transformed = [construct_prompt(sentence) for sentence in X]

predicted_labels = []
# predicted_labels = [classify_sentence(sentence) for sentence in X_transformed]

for i in tqdm(range(len(X_transformed))):
    # print(X_transformed[i])
    p = classify_sentence(X_transformed[i])
    predicted_labels.append(p)
    # print(p)
    time.sleep(0.13)
    # print("------")

pred_labels = [1 if 'true' in label.lower() else 0 for label in predicted_labels]

# print(X_transformed)

correct_predictions = np.sum(np.array(pred_labels) == np.array(y))
accuracy = correct_predictions / len(y)
print(f"Accuracy: {accuracy * 100:.2f}%")


confmap = sns.heatmap(confusion_matrix(y, pred_labels),annot=True);
fig = confmap.get_figure()
fig.savefig("confusion_matrix.png") 
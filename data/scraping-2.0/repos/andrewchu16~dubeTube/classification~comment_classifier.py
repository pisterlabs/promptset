import os
import cohere
from cohere.classify import Example

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
EXAMPLE_FILE = "classification/comments.txt"


def classify_comment(comment):

    co = cohere.Client(API_KEY)

    prompts = []
    with open(EXAMPLE_FILE, "r") as example_file:
        lines = example_file.readlines()
        for i in range(0, len(lines), 2):
            prompts.append(Example(lines[i].rstrip(), lines[i+1].rstrip()))
    
    classifications = co.classify(
        model = "medium",
        inputs=[comment],
        examples=prompts
    )

    item_one = classifications.classifications[0].confidence[0]
    item_two = classifications.classifications[0].confidence[1]
    if item_one.confidence > item_two.confidence:
        return item_one.label
    else:
        return item_two.label


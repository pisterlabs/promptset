import os
import pickle

import openai
from dotenv import load_dotenv

from gptme.conversation import Conversation, Message
from gptme.utils.semantic_search import semantic_search
from transformers import pipeline, QuestionAnsweringPipeline

qa: QuestionAnsweringPipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

load_dotenv(dotenv_path=".env")

openai.api_key = os.environ["OPENAI_KEY"]

with open(
    ".memories/1684566721.4356935/messages_1055763501693554709_vinq#6132.txt.pickle",
    "rb",
) as f:
    embeddings, transcript = pickle.load(f)

print(len(embeddings), len(transcript))

"What is Katie's topic for the research report?"

while True:
    question = input("Question > ")

    results = list(
        semantic_search(
            query=question,
            embeddings=embeddings,
            transcript=transcript,
        )
    )

    for index, result in enumerate(results):
        print(f"---- Result #{index+1} ----")
        print(result[0])
        print(f"Chance: {result[1]}")

        answer = qa(question=question, context=result[0], top_k=3)

        print(f"Answer: {answer}")

    # conversation = Conversation(
    #     messages=[
    #         Message(
    #             content="Answer the question in a single complete sentence.",
    #             role="system",
    #         ),
    #         Message(content=result[0], role="user"),
    #         Message(content=question, role="user"),
    #     ]
    # )

    # summary = conversation.get_completion_chat()

    # print(summary)

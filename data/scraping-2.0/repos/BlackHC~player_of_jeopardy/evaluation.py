from dataclasses import dataclass
from typing import List, Optional

import datasets
import langchain
import numpy as np
import pandas as pd
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, AIMessage, ChatGeneration, ChatResult, Generation
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm

# %%
# Load the dataset
dataset = datasets.load_dataset("jeopardy")

# Note: there is exactly one split. You can list all the splits with `dataset.keys()`
dataset = dataset["train"]

# %%
# Print the first 10 rows
print(dataset.select(range(10)).data)


# %%

# Create a langchain prompt for the dataset.
# We want to play Jeopardy, so we want to ask questions and get answers.
# We want to include the category and the question in the prompt.


@dataclass
class JeopardyResults:
    idx: object
    question: str
    category: str
    true_answer: str
    model_answer: str
    accuracy: float
    confidence: float


def evaluate_jeopardy_row(chat_model: ChatOpenAI, idx: object, question: str, category: str, true_answer: str):
    chat_chain = [
        SystemMessage(content="You are playing Jeopardy. You are the contestant. You are answering questions."),
        HumanMessage(content=f"Do you know the game Jeopardy?"),
        AIMessage(content=
                  "Yes, Jeopardy! is a popular American television game show that has been on the air since "
                  "1964. In the game, contestants are presented with answers to trivia questions in various "
                  "categories, and they must respond with a question that corresponds to the answer. "
                  "The show is known for its unique format, where the answers are presented first, and the contestants must "
                  "phrase their responses in the form of a question. The show has become a cultural phenomenon and has been "
                  "adapted in many countries around the world."
                  ),
        HumanMessage(content=f"Let's play.\n\nCategory: {category}\n{question}\n"),
    ]
    model_answer = chat_model(chat_chain)
    assert isinstance(model_answer, AIMessage)
    chat_chain.append(model_answer)

    # How confident are we in the answer?
    confidence_chain = list(chat_chain)
    confidence_chain.append(HumanMessage(
        #content="Are you sure? Give a short reasoning and then please give a confidence between 0 and 1 about how certain you are this is the correct answer."
        content="Please give a confidence between 0 and 1 about how certain you are this is the correct answer."
    ))
    confidence_reason = chat_model(confidence_chain)
    assert isinstance(confidence_reason, AIMessage)
    confidence_chain.append(confidence_reason)

    for i in range(3):
        confidence_chain.append(HumanMessage(
            content="Please only reply with the number and just that. Example: \"0.90.\" or \"0.5.\". I need to parse your response."
        ))
        confidence = chat_model(confidence_chain)
        assert isinstance(confidence, AIMessage)

        try:
            confidence = float(confidence.content.rstrip("."))  # Remove trailing dot.
            break
        except ValueError:
            pass

        # Also try to split of the first word.
        try:
            confidence_value = float(confidence.content.split()[0].rstrip("."))  # Remove trailing dot.
            print(f"Split confidence: {confidence_value} from {confidence.content}")
            confidence = confidence_value
            break
        except ValueError:
            pass

        # Also try to split off the last word.
        try:
            confidence_value = float(confidence.content.split()[-1].rstrip("."))  # Remove trailing dot.
            print(f"Split confidence: {confidence_value} from {confidence.content}")
            confidence = confidence_value
            break
        except ValueError:
            pass

        confidence_chain.append(confidence)
    else:
        print(f"Expected confidence between 0 and 1, got {confidence}")
        confidence = float("nan")

    if confidence != float("nan") and confidence < 0 or confidence > 1:
        print(f"Expected confidence between 0 and 1, got {confidence}")
        confidence = float("nan")

    # Verify the answer.
    accuracy_chain = list(chat_chain)

    accuracy_chain.append(HumanMessage(
        content=f"Let's verify. The solution book says, it's {true_answer}. Does this match your solution above? "
                "Only answer yes or no. Example: 'Yes.' or 'No.'")
    )
    model_evaluation = chat_model(accuracy_chain)

    # Verify the evaluation.
    for i in range(2):
        yes_or_no = model_evaluation.content.lower().strip().rstrip(".")
        if yes_or_no in ["yes", "no"]:
            break
        accuracy_chain.append(model_evaluation)
        accuracy_chain.append(HumanMessage(content=f"Please only answer with 'Yes.' or 'No.'! I need to parse the answer."))
        model_evaluation = chat_model(accuracy_chain)

    assert yes_or_no in ["yes", "no"], f"Expected yes or no, got {yes_or_no}"
    if yes_or_no == "yes":
        accuracy = 1
    else:
        accuracy = 0

    return JeopardyResults(
        idx=idx,
        question=question,
        category=category,
        true_answer=true_answer,
        model_answer=model_answer.content,
        accuracy=accuracy,
        confidence=confidence,
    )


# %%


langchain.llm_cache = SQLiteCache(".chat.langchain.db")


class CachedChatOpenAI(ChatOpenAI):
    """
    A chat model that caches the results of the LLM and uses ChatOpenAI as the base.
    """

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        messages_prompt = repr(messages)
        if langchain.llm_cache:
            results = langchain.llm_cache.lookup(messages_prompt, self.model_name)
            if results:
                assert len(results) == 1
                result: Generation = results[0]
                chat_result = ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=result.text))],
                    llm_output=result.generation_info,
                )
                return chat_result
            raise ValueError("No results found in cache.")
        chat_result = super()._generate(messages, stop)
        if langchain.llm_cache:
            assert len(chat_result.generations) == 1
            result = Generation(text=chat_result.generations[0].message.content, generation_info=chat_result.llm_output)
            langchain.llm_cache.update(messages_prompt, self.model_name, [result])
        return chat_result


chat_model = CachedChatOpenAI(max_tokens=512, )

# %%
# Set seed to make the results reproducible.
np.random.seed(42)

# Randomly sample 100 indices from the dataset.
n = 20000
indices = np.random.choice(len(dataset), n, replace=False)

# %%

# Evaluate the dataset.
results = []

# Only take the first 2 rows
num_correct = 0
for i, idx in enumerate(tqdm(indices)):
    idx = int(idx)
    row = dataset[idx]
    result = evaluate_jeopardy_row(chat_model, idx, row["question"], row["category"], row["answer"])
    num_correct += result.accuracy
    results.append(result)

    # Every 30 rows, print the accuracy.
    if i != 0 and i % 30 == 0:
        print(f"Accuracy: {num_correct / len(results)}")


# %%
print(f"Accuracy: {num_correct / len(results)}")

# %%

df = pd.DataFrame(results)

# Make idx the index
df.set_index("idx", inplace=True)

# Save as CSV
df.to_csv("jeopardy_results.csv", index=True)
np_df = df.to_numpy()
# save numpy
np.save('jeopardy_results.npy', np_df)

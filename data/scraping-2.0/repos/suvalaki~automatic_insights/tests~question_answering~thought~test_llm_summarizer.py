import pytest


from ai.question_answering.thought.base import Thought
from ai.question_answering.thought.llm_comparison import LLThoughtPairComparer
from ai.question_answering.thought.llm_summarizer import LLMThoughtSummarizer


SHARED = """
Sleep should be at least 4 hours per day. 
Mobile phones are quite expensive, as the amount spent on them has increased over time. 
"""

UNIQUE_TO_1 = """
The Sky is green. 
The grass is blue.
Drinking tea gives you a flu.
"""


UNIQUE_TO_2 = """
Apples are a fruit.
There can only ever be one pope at a time. 
Sleep is an important part of your day.
"""

CONTRADICTORY_IN_1 = """
Glasses help you see better only when they have square frames. 
Better microphones don't make you sound better over the phone.
"""

CONTRADICTORY_IN_2 = """
Better microphones are an important part of having higher quality phone calls.
Better microphones will be appreciated by listeners. 
Glasses only help you see better when you arent wearing them.
"""

UNIQUE_TO_3 = """
Grey is a shade and not a color. 
The speed of light is constant.
"""

UNIQUE_TO_4 = """
Housing is a need and not a want. 
Community standards are improving over time.
"""


def test_llm_thoughts_summarizer():
    from langchain.chat_models import ChatOpenAI

    model_name = "gpt-4"
    temperature = 0.0
    model = ChatOpenAI(model_name=model_name, temperature=temperature)

    thought0 = Thought(discussion=SHARED + UNIQUE_TO_1 + CONTRADICTORY_IN_1, score=0.0)
    thought1 = Thought(discussion=SHARED + UNIQUE_TO_2 + CONTRADICTORY_IN_2, score=0.0)
    thought2 = Thought(discussion=UNIQUE_TO_3, score=0.0)
    thought3 = Thought(discussion=UNIQUE_TO_4, score=0.0)

    # Thought0 and 1 are contradictory and so these should be filtered out
    # Thought2 is unique and so should be retained.

    comparer = LLThoughtPairComparer(model)
    summarizer = LLMThoughtSummarizer(model, comparer)

    summary = summarizer.summarize([thought0, thought1, thought2, thought3])
    print(summary.discussion, summary.score)
    print(summary)

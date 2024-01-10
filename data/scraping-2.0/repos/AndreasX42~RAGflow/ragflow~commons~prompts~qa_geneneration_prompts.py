# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate

temp_chat_1 = """You are a smart assistant that can identify key information in a given text and extract a corresponding question and answer pair of this key information so that we can build a Frequently Asked Questions (FAQ) section. Both the question and the answer should be comprehensive, well formulated and readable for a broad and diverse audience of readers.

When coming up with this question/answer pair, you must respond in the following format:

```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

Everything between the ``` must be valid json.
"""
temp_chat_2 = """Please come up with a question and answer pair, in the specified JSON format, for the following text:
----------------
{text}"""

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(temp_chat_1),
        HumanMessagePromptTemplate.from_template(temp_chat_2),
    ]
)

temp_dft = """You are a smart assistant that can identify key information in a given text and extract a corresponding question and answer pair of this key information so that we can build a Frequently Asked Questions (FAQ) section. Both the question and the answer should be comprehensive, well formulated and readable for a broad and diverse audience of readers.

When coming up with this question/answer pair, you must respond in the following format:

```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

Everything between the ``` must be valid json.

Please come up with a question/answer pair, in the specified JSON format, for the following text:
----------------
{text}
"""

PROMPT = PromptTemplate.from_template(temp_dft)

QA_GENERATION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

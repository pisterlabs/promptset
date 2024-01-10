import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

ENCODING = tiktoken.get_encoding("cl100k_base")
SUMMARIZE_MODEL = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.2)
MAX_TOKENS_SUMMARY = 3000
SUMMARY_SYS_MSG = """You are SummaryGPT, a model designed to ingest content and summarize it concisely and accurately.
You will receive an input string, and your response will be a summary of this information."""


def token_len(input: str) -> int:
    """Get token length for openai"""
    return len(ENCODING.encode(input))

def chunk(input: str) -> list:
    input_tokens = token_len(input)
    count = math.ceil(input_tokens / MAX_TOKENS_SUMMARY)
    k, m = divmod(len(input), count)
    chunks = [
        input[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(count)
    ]
    return chunks

def summarize(input: str) -> str:
    system_message = SystemMessagePromptTemplate.from_template(
        template=SUMMARY_SYS_MSG
    )
    human_message = HumanMessagePromptTemplate.from_template(
        template="Input: {input}"
    )

    chunks = chunk(input=input)

    summary = ""

    for i in chunks:
        prompt = ChatPromptTemplate(
            input_variables=["input"],
            messages=[system_message, human_message],
        )

        _input = prompt.format_prompt(input=i)
        output = SUMMARIZE_MODEL(_input.to_messages())
        summary += f"\n{output.content}"

    sum_tokens = token_len(input=summary)

    if sum_tokens > MAX_TOKENS_SUMMARY:
        return summarize(input=summary)

    return summary
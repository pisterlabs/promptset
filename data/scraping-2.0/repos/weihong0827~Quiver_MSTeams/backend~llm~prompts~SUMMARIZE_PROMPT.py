from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and some follow up response,the follow up responses are meant to be accepted answers for the questions that is asked previously, summarize them to extract the main question and answer of this conversation, in its original language.

Chat History:
{chat_history}
Follow Up Input: {response}
summary:"""
SUMMARIZE_PROMPT = PromptTemplate.from_template(_template)

from langchain.prompts.prompt import PromptTemplate

_template = """次の会話と追加の質問があった場合、日本語で追加の質問に答えてください。与えられた文脈で正しい答えがない場合は、自分の知識に基づいて答えてみてください。答えがわからない場合は、答えをでっち上げようとせず、わからないと言ってください。日本語での回答が難しい場合は英語でも構いません。

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """次の文脈を使用して、日本語で質問に答えてください。与えられた文脈で正しい答えがない場合は、自分の知識に基づいて答えてみてください。答えがわからない場合は、答えをでっち上げようとせず、わからないと言ってください。日本語での回答が難しい場合は英語でも構いません。

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from state_store import State, get_state


class QuestionContext:
    def __init__(self, repo_name, github_url, extension_freqs):
        self.repo_name = repo_name
        self.github_url = github_url
        self.extension_freqs = extension_freqs


def form_prompt_template():
    template = """
        Repo: {repo_name} ({github_url}) | Q: {question} | FileCount: {extension_freqs} | Docs: {relevant_documents}

        Instr:
        1. Answer based on context/docs.
        2. Focus on repo/code.
        3. Consider:
            a. Purpose/features - describe.
            b. Functions/code - give details/samples.
            c. Setup/usage - give instructions.

        Answer:
    """
    prompt_template = PromptTemplate(
        template=template,
        input_variables=[
            "repo_name",
            "github_url",
            "question",
            "extension_freqs",
            "relevant_documents",
        ],
    )
    return prompt_template


def get_answer(question, context: QuestionContext, retriever):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    llm_chain = LLMChain(prompt=form_prompt_template(), llm=llm)

    relevant_documents = retriever.invoke(question)
    phrased_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{relevant_documents}"

    answer = llm_chain.run(
        question=question,
        context=phrased_context,
        repo_name=context.repo_name,
        github_url=context.github_url,
        extension_freqs=context.extension_freqs,
        relevant_documents=relevant_documents,
    )
    return answer

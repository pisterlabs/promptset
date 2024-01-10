from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

from arcan.agent.llm import LLM
from arcan.agent.templates import chains_templates


class ArcanConversationChain:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.llm = LLM().llm
        self.embeddings = OpenAIEmbeddings()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def set_chain(self, **kwargs):
        condense_question_prompt = PromptTemplate.from_template(
            chains_templates()["chat_chain_template"]
        )
        QA_PROMPT = PromptTemplate(
            template=chains_templates()["chat_prompt_template"],
            input_variables=["question", "context"],
        )

        question_generator = LLMChain(llm=self.llm, prompt=condense_question_prompt)

        doc_chain = load_qa_chain(
            llm=self.llm,
            chain_type=kwargs.get(
                "chain_type", "stuff"
            ),  # Should be one of "stuff","map_reduce", "map_rerank", and "refine".
            prompt=QA_PROMPT,
        )
        return question_generator, doc_chain

    def get_qa_retrieval_chain(self, vectorstore):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.kwargs.get("chain_type", "stuff"),
            retriever=vectorstore.as_retriever(),
        )

    def get_chat(self, vectorstore):
        question_generator, doc_chain = self.set_chain()
        return ConversationalRetrievalChain(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
        )

    def run(self, prompt, vectorstore):
        chain = self.get_chat(vectorstore)
        try:
            with get_openai_callback() as cb:
                return self.run_with_openai_callback(chain, prompt, cb)
        except Exception as e:
            print(e)
            return chain.run(prompt)

    def run_with_openai_callback(self, chain, prompt, cb):
        result = chain.run(prompt)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return result


def retrieve_sources(sources_refs: str, texts: list[str]) -> list[str]:
    """
    Map back from the references given by the LLM's output to the original text parts.
    """
    clean_indices = [r.replace("-pl", "").strip() for r in sources_refs.split(",")]
    numeric_indices = (int(r) if r.isnumeric() else None for r in clean_indices)
    return [texts[i] if i is not None else "INVALID SOURCE" for i in numeric_indices]

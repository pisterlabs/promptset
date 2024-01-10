from langchain import LLMChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from .prompts import QAPrompt, ChatPrompt
from .base import LlmConfig


class QaLlmApp:
    def __init__(self, llm, vectordb, prompt=QAPrompt().qa_prompt) -> None:
        self.cfg = LlmConfig()
        self.llm_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vectordb.as_retriever(search_kwargs={'k': self.cfg.VECTOR_COUNT}),
            return_source_documents=self.cfg.RETURN_SOURCE_DOCUMENTS,
            chain_type_kwargs={'prompt': prompt}
        )

    def __call__(self, query):
        return self.llm_qa({'query': query})


class ChatLlmApp:
    def __init__(self, llm, prompt=ChatPrompt().chat_prompt, memory=ConversationBufferWindowMemory(), verbose=True) -> None:
        self.llm_chat = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=verbose,
            memory=memory
        )

    def __call__(self, inputs) -> str:
        return self.llm_chat.predict(human_input=inputs)

from __future__ import annotations

from typing import Iterable, List, Optional, Type, Union

from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
import torch

from llm.inference import TransformersEngine, InferenceEngine
from llm.model_configs import ModelConfig
from llm.prompt import Message, Prompt, Conversation
from llm.qa.prompts import FewShotQA, StandaloneQuestion


class QASession:
    """
    Manages session state for a question-answering conversation between a user and an AI.
    Contexts relevant to questions are retrieved from the given vector store and appended
    to the system prompt that is fed into inference.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        vector_store: VectorStore,
        qa_prompt: Prompt,
        rephrase_prompt: Prompt,
        conversation: Optional[Conversation] = None,
        debug: bool = False,
    ):
        self.engine = engine
        self.vector_store = vector_store
        self.conversation = conversation if conversation else Conversation()
        self.qa_prompt = qa_prompt
        self.rephrase_prompt = rephrase_prompt
        self.debug = debug
        self.results: List[Document] = []
        self.contexts: List[str] = []

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig,
        vector_store: VectorStore,
        engine: Optional[InferenceEngine] = None,
        qa_prompt: Union[Prompt, Type[Prompt]] = FewShotQA,
        rephrase_prompt: Union[Prompt, Type[Prompt]] = StandaloneQuestion,
        **kwargs,
    ) -> QASession:
        if engine is None:
            engine = TransformersEngine.from_model_config(
                model_config,
                load_kwargs={
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "quantization_config": "4bit",
                },
            )
        if isinstance(qa_prompt, type):
            qa_prompt = qa_prompt.from_model_config(model_config)
        if isinstance(rephrase_prompt, type):
            rephrase_prompt = rephrase_prompt.from_model_config(model_config)
        return cls(
            engine, vector_store, qa_prompt=qa_prompt, rephrase_prompt=rephrase_prompt, **kwargs
        )

    def stream_answer(self, question: str, update_context: bool = False, **kwargs) -> Iterable[str]:
        """
        Stream response to the given question using the session's prompt and contexts.
        """
        if update_context:
            self.search_context(question)

        last_message = self.last_message
        if last_message and last_message.input == question and last_message.response is None:
            # Message already added to conversation
            message = self.last_message
            message.contexts = self.contexts
        else:
            message = self.append_question(question, contexts=self.contexts)

        input_text = self.conversation.render(self.qa_prompt)
        if self.debug:
            print(f"\n** Context Input **\n{input_text}")

        gen_kwargs = {
            "stop_strings": self.qa_prompt.stop_strings,
            **kwargs,
        }

        output_text = ""
        for output_text in self.engine.generate_stream(input_text, **gen_kwargs):
            output_text = output_text.strip()
            message.response = output_text
            yield output_text

        if self.debug:
            print(f"\n** Context Answer **\n{output_text}")

    def rephrase_question(self, question: str, **kwargs):
        """
        Rephrase question to be a standalone question based on conversation history.

        Enables users to implicitly refer to previous messages. Relevant information is
        added to the question, which then gets used for semantic search of contexts.
        """
        last_message = self.last_message
        if not last_message:
            # No history to use for rephrasing
            return question

        if last_message.input == question and last_message.response is None:
            # Question already added to conversation
            messages = self.conversation.messages[:-1]
            if len(messages) == 0:
                # No history to use for rephrasing
                return question
        else:
            messages = self.conversation.messages

        contexts: List[str] = []
        for m in messages:
            contexts.extend((m.input, m.response or ""))
        message = Message(question, contexts=contexts)

        input_text = self.rephrase_prompt.render(messages=[message])
        if self.debug:
            print(f"\n** Standalone Input **\n{input_text}")

        params = {
            "stop_strings": self.rephrase_prompt.stop_strings,
            **kwargs,
        }
        standalone = self.engine.generate(input_text, **params).strip()
        if self.debug:
            print(f"\n** Standalone Question **\n{standalone}")
        return standalone

    def append_question(self, question: str, **kwargs) -> Message:
        message = Message(input=question, **kwargs)
        self.conversation.add(message)
        return message

    def search_context(self, question: str, top_k: int = 3, **kwargs) -> List[Document]:
        """
        Update contexts from vector store
        """
        self.results = self.vector_store.similarity_search(question, top_k, **kwargs)
        self.set_contexts([r.page_content for r in self.results])
        return self.results

    def set_contexts(self, contexts: List[str]):
        """
        Set contexts explicitly (e.g. for filtering which results are included)
        """
        self.contexts = contexts

    @property
    def has_history(self) -> bool:
        return len(self.conversation.messages) > 0

    @property
    def last_message(self) -> Optional[Message]:
        if self.has_history:
            return self.conversation.messages[-1]
        return None

    def get_history(
        self,
        user_label: str = "Question: ",
        assistant_label: str = "Answer: ",
        separator: str = "\n",
    ) -> str:
        """
        Get conversation history formatted as a string
        """
        history = []
        for message in self.conversation.messages:
            history.append(f"{user_label}{message.input}")
            if message.response is not None:
                history.append(f"{assistant_label}{message.response}")

        return separator.join(history)

    def clear(self, keep_results: bool = False):
        self.conversation.clear()
        if not keep_results:
            self.results = []
            self.contexts = []

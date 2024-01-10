from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import CHAT_TURN_TYPE, \
    _ROLE_MAP
from langchain.schema import BaseMessage
from typing import Dict, Any, Optional, List
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun


def _get_chat_history(chat_history: List[CHAT_TURN_TYPE]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type,
                                        f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, list) or isinstance(dialogue_turn,
                                                           tuple):
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer


class MyConversationalRetrievalChain(ConversationalRetrievalChain):

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()

        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str,
                callbacks=callbacks
            )
        else:
            new_question = question

        # get any additional kwargs for the vector store
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        # this method returns both the documents and the similarity score - how relevant the document is to the question
        # and we can present this score to the user as "Source confidence" - which would be a major feature of the chatbot
        # retriever.vectorstore
        sources_with_similarity = self.retriever.vectorstore.similarity_search_with_score(
            new_question, k=self.max_tokens_limit, **vectordbkwargs
        )

        # let's return these confidences to the user and present them as "Source confidence"
        sources = []
        for doc, similarity in sources_with_similarity:
            doc.metadata['confidence'] = round((1 - similarity) * 100, 2)
            sources.append(doc)

        new_inputs = inputs.copy()

        new_inputs["question"] = question

        new_inputs["chat_history"] = chat_history_str

        answer, _ = await self.combine_docs_chain.acombine_docs(
            sources,
            callbacks=_run_manager.get_child(),
            **new_inputs
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": sources}
        else:
            return {self.output_key: answer}

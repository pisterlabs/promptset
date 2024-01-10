from langchain.memory.buffer import ConversationBufferMemory
from chainlink.prompts_mem import (
    POTENTIAL_ANSWER_PROMPT,
    FINAL_ANSWER_PROMPT,
    FINAL_ANSWER_PROMPT_2,
    VERIFICATION_PROMPT,
    QUESTION_MODIFIER_PROMPT,
)
from chainlink.utils import (
    prepare_single_document,
    prepare_multiple_documents,
    retriever,
    chain,
)
from utils import createLogHandler

logger = createLogHandler(__name__, "logs.log")


def get_answer_memory(question, memory):
    if memory.chat_memory.messages:
        chain.prompt = QUESTION_MODIFIER_PROMPT
        question = chain.predict(question=question, history=memory.buffer)
        logger.info(f"Modified question: {question}")

    retrieved_docs = retriever.get_relevant_documents(question)
    all_answers = []
    for i, d in enumerate(retrieved_docs):
        chain.prompt = POTENTIAL_ANSWER_PROMPT
        response = chain.predict(question=question, document=prepare_single_document(d))
        all_answers.append(
            {"id": i, "answer": response, "original_source": d.metadata["source"]}
        )

    result = prepare_multiple_documents(all_answers)

    chain.prompt = FINAL_ANSWER_PROMPT
    answer = chain.predict(question=question, document=result, history=memory.buffer)

    if any(
        answer.lower().startswith(x)
        for x in ["i don't know", "i dont know", "i do not know", "i don't know."]
    ):
        pot_answer = ""
        for i, d in enumerate(retrieved_docs):
            if not pot_answer:
                chain.prompt = FINAL_ANSWER_PROMPT_2
                answer = chain.predict(
                    question=question, document=result, history=memory.buffer
                )
                if not any(
                    answer.lower().startswith(x)
                    for x in [
                        "i don't know",
                        "i dont know",
                        "i do not know",
                        "i don't know.",
                    ]
                ):
                    pot_answer = answer
                    break
        if not pot_answer:
            answer = "Sorry, I don't know the answer to that question."
        else:
            answer = pot_answer

    memory.save_context({"question": question}, {"answer": answer})

    return answer, memory

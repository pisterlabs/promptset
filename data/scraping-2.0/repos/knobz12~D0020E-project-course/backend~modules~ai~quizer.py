"""
Creating quizes based on document(s)

# Improvements:
* Amount of quiz answers per question
"""

import json
from modules.ai.utils.llm import create_llm_guidance
from modules.ai.utils.vectorstore import *

import guidance
from guidance import select, gen

from modules.files.chunks import *

from typing import Any, Generator

def calculate_questions_per_doc(total_docs: int, total_questions: int, doc_index: int):
    """
    Calculate the number of questions to generate for a specific document.

    Parameters:
    - total_docs (int): Total number of documents.
    - total_questions (int): Total number of questions to generate.
    - doc_index (int): Index of the current document (0-based).

    Returns:
    - int: Number of questions to generate for the specified document.
    """
    # Ensure that the inputs are valid
    if total_docs <= 0 or total_questions <= 0 or doc_index < 0 or doc_index >= total_docs:
        raise ValueError("Invalid input parameters")

    # Calculate the base number of questions for each document
    base_questions_per_doc = total_questions // total_docs

    # Calculate the remaining questions after distributing the base questions
    remaining_questions = total_questions % total_docs

    # Calculate the actual number of questions for the current document
    questions_for_current_doc = base_questions_per_doc + (1 if doc_index < remaining_questions else 0)

    return questions_for_current_doc

@guidance()
def determineQuestionBias(lm, question: str):
    lm += f"""I want you to determine if a question for a quiz is factual or opinion based.
    For example a question about your opinion about something would be opinion based and a question about a fact is factual.

    Question: {question}

    Answer: {select(options=["factual", "opinion"])}"""
    return lm


@guidance()
def newQuestionJSONGenerator(lm, context: str, answer_count: int, question_count: int):

    def gen_answer(idx: int,questionIndex:int) -> str:
        answerKey = f"answer{questionIndex}-{idx}"
        isAnswerKey = f"isAnswer{questionIndex}-{idx}"
        print(answerKey, isAnswerKey)
        import random
        seed = random.randint(0, 1337)
        answer: str = f"""\"{gen(name=answerKey, stop='"',llm_kwargs={"seed": seed})}\": {select(["True", "False"],name=isAnswerKey)}"""
        return answer

    def gen_question(idx: int):
        # question: str = f"""{{ "question":"{gen(f"question{idx}",stop='"',)}", "answers":["""
        question: str = f"""\
Question: "{gen(f"question{idx}",stop='"')}"
Answers:
"""

        # answers: str = ""
        for i in range(0, answer_count):
            question += gen_answer(i, idx) + "\n"
        

        #print(question)
        return question

    questions: str = ""
    for i in range(0, question_count):
        questions += gen_question(i) + "\n\n"

        
    print("Questions:\n", questions)




    res = f"""\
The following is a quiz question in JSON format.
Generate answers based on the provided context. Only ONE of the answers is. true, and the others shall be false.
The incorrect answers must be different from each other but still related to the topic.
The questions MUST be different to one another.

Context: {context}

Questions:

{questions}
    """


    print(res)
    lm += res 
    
    return lm








def create_quiz(id: str, questions: int) -> str:
    glmm = create_llm_guidance()
    vectorstore = create_vectorstore()

    docs = vectorstore.get(limit=100,include=["metadatas"],where={"id":id})
    print(docs)


    obj: dict[str, list[dict[str, Any]]] =  {}

    for (i, doc) in enumerate(docs["metadatas"]):
        qsts_cunt = calculate_questions_per_doc(len(docs["metadatas"]), questions, i)
        print(f"Questions for {i}")
        result = glmm + newQuestionJSONGenerator(doc["text"], 4, qsts_cunt)
        print(str(result))

        obj["questions"] = []

        for i in range(0, qsts_cunt):
            question: str = result[f"question{i}"]
            obj["questions"].append({"question" : question, "answers": []})

            for j in range(0,4):
                answer: str = result[f"answer{i}-{j}"]
                correct: str = result[f"isAnswer{i}-{j}"]
                obj["questions"][i]["answers"].append({"text": answer, "correct" : False if correct == "False" else True})
    
    result: str = json.dumps(obj)
    return result



def quiz_test():
    file_hash = Chunkerizer.upload_chunks_from_file("backend/tests/sample_files/Test_htmls/Architectural Design Patterns.html", "D0072E")
    print(create_quiz(file_hash, 3)) 

if __name__ == "__main__":
    quiz_test()
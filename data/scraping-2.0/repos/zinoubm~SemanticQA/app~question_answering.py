from openai_manager.core import ask, summarize
from openai_manager.base import OpenAiManager
from db.datastore import QdrantManager
from models.models import Question

openai_manager = OpenAiManager()
db_manager = QdrantManager()


def answer_question(question: Question):
    question_embedding = openai_manager.get_embedding("what are my rights?")
    scored_answer_chunks = db_manager.search_point(question_embedding, 3)

    answer_chunks = [answer.payload["chunk"] for answer in scored_answer_chunks]

    return ask("\n".join(answer_chunks), question.question, openai_manager)


if __name__ == "__main__":

    res = answer_question(Question(question="what are my rights?"))
    print(res)

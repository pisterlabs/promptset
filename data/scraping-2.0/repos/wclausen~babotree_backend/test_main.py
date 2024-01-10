import random
import timeit

import openai

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, ContentEmbedding
from main import get_openai_summary, _get_readwise_highlights


def test_get_readwise_articles():
    print()
    db = get_direct_db()
    articles = _get_readwise_highlights(db)
    print(articles)
def test_get_readwise_highlights_timing():
    print("Timeit results:")
    print(timeit.timeit(test_get_readwise_articles, number=10) / 10)

openai.api_key = babotree_utils.get_secret('OPENAI_API_KEY')
def get_llm_response(highlights, prompt):
    openai_question_function_tools = [{
        "type": "function",
        "function": {
            "name": "create_question_answer_pair",
            "description": "Creates a question answer pair",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question string",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The correct answer to the question",
                    },
                },
                "required": ["question", "answer"]
            },
        }
    },
    ]
    messages = [
        {
            "role": "system",
            "content": "You are an expert educator, a master of helping students traverse Bloom's taxonomy and understand subjects on a deep level.",
        },
        {
            "role": "user",
            "content": prompt + "\n---\n" + "\n".join(
                [highlight.text for highlight in highlights]),
        },
    ]
    print("Fetching questions from openai")
    print(messages)
    response = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=1.0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.45,
    )
    print(response.choices[0].message.content)
def test_prompts_for_generating_questions():
    possible_prompts = [
        "Consider the following excerpts from a book called Docker Deep Dive. Formulate a question from these excepts that tests the readers knowledge. The question should be challenging but solvable given the information in the excerpts:",
        "Consider the following excerpts from a book called Docker Deep Dive. Formulate a question/answer pair from these excepts that tests the readers knowledge. The question should test an atomic unit of information, but should not be a True/False question:"
    ]
    db = get_direct_db()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id == '0a16c1a1-33b3-4777-8d2a-59347d1a985a').all()
    print(docker_highlights)
    get_llm_response(random.choices(docker_highlights, k=3), possible_prompts[1])

def test_embedding_highlights_similarity():
    db = get_direct_db()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id == '0a16c1a1-33b3-4777-8d2a-59347d1a985a').all()
    print()
    for docker_highlight in docker_highlights:
        print(docker_highlight.text)
        docker_highlight_embedding = db.query(ContentEmbedding).filter(ContentEmbedding.source_id == docker_highlight.id, ContentEmbedding.source_type == 'HIGHLIGHT_TEXT').first()
        # print(docker_highlight_embedding)
        # print()
        closest_highlights = db.query(Highlight).join(ContentEmbedding, ContentEmbedding.source_id == Highlight.id).filter(ContentEmbedding.source_type == 'HIGHLIGHT_TEXT', Highlight.id != docker_highlight.id).order_by(ContentEmbedding.embedding.cosine_distance(docker_highlight_embedding.embedding)).limit(3).all()
        print("Nearest neighbors")
        print("\n".join([x.text for x in closest_highlights]))
        print("LLM question")
        get_llm_response([docker_highlight] + closest_highlights, "Consider the following excerpts from a book called Docker Deep Dive. Formulate a question/answer pair from these excepts that tests the readers knowledge. The question should test an atomic unit of information, but should not be a True/False question:")
        print('----')



def test_get_openai_summary():
    print()
    source_id = 5561801
    response = get_openai_summary(source_id)
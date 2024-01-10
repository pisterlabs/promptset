from __future__ import annotations
from dotenv import load_dotenv

import pandas as pd
import openai
import numpy as np
import scipy as sci

import os

load_dotenv('.env')

openai.api_key = os.environ["OPENAI_ACCESS_TOKEN"]

COMPLETIONS_MODEL = "text-davinci-003"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"
DOC_EMBEDDINGS_MODEL = f"text-embedding-ada-002"
QUERY_EMBEDDINGS_MODEL = f"text-embedding-ada-002"


MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 150,
    "model": COMPLETIONS_MODEL,
}

class Question:
    def __init__(self, question, answer, context, audio_src_url) -> None:
        self.pk = 1
        self.question = question
        self.answer = answer
        self.context = context
        self.audio_src_url = audio_src_url
        self.ask_count = 1
        pass

class ExampleQA:
    def __init__(self, header, QA_list) -> None:
        self.header = header
        self.QA_list = QA_list
    
    def merged_QA(self) -> str:
        return "".join(self.QA_list)

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    but very similiar because x and y embedding vector nomalization is close to 1., so that cos divided nomalization equals to 1.
    cosine similarity: 1 - cosine_distance = 1- sci.spatial.distance.cosine(np.array(x), np.array(y)). 
    note: distinguish between distance and similarity.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def load_embeddings(fname: str) -> dict[str, list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    df.dropna(axis='columns', how='all', inplace=True)
    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, example_QA: ExampleQA) -> tuple[str, str]:
    """
    Fetch relevant embeddings
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[df['title'] == section_index].iloc[0]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            space_left = MAX_SECTION_LEN - chosen_sections_len - len(SEPARATOR)
            chosen_sections.append(SEPARATOR + document_section.content[:space_left])
            chosen_sections_indexes.append(str(section_index))
            break

        chosen_sections.append(SEPARATOR + document_section.content)
        chosen_sections_indexes.append(str(section_index))

    header = example_QA.header
    
    return (header + "".join(chosen_sections) + question_1 + question_2 + question_3 + question_4 + question_5 + question_6 + question_7 + question_8 + question_9 + question_10
             + "\n\n\nQ: " + question + "\n\nA: "), ("".join(chosen_sections))

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    example_QA,
) -> tuple[str, str]:
    prompt, context = construct_prompt(
        query,
        document_embeddings,
        df,
        example_QA=example_QA
    )

    print("===\n", prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n"), context


def ask(request, book_pages_path, example_QA):
    question_asked = request.get("question", "")

    if not question_asked.endswith('?'):
        question_asked += '?'

    # previous_question = Question.objects.filter(question=question_asked).first()
    previous_question = None
    previous_question = Question("How do I decide what kind of business I should start?",
                                  "There's no easy answer to this question, as it depends on a variety of factors including your skills, interests, and goals. However, a good place to start is by considering what kind of problem you want to solve or what need you want to fill. Once you have a general idea of the kind of business you want to start, you can begin researching specific",
                                  "what aspects and idea do you have? look at your around and find what problem your friends distressed in.",
                                  "audio11")
    audio_src_url = previous_question and previous_question.audio_src_url
    audio_src_url = None

    if audio_src_url:
        print("previously asked and answered: " + previous_question.answer + " ( " + previous_question.audio_src_url + ")")
        previous_question.ask_count = previous_question.ask_count + 1
        return { "question": previous_question.question, "answer": previous_question.answer, "audio_src_url": audio_src_url, "id": previous_question.pk }

    df = pd.read_csv(book_pages_path)
    embedding_path = book_pages_path.replace(".pdf.pages.csv", ".pdf.embeddings.csv")
    document_embeddings = load_embeddings(embedding_path)
    answer, context = answer_query_with_context(question_asked, df, document_embeddings, example_QA)

    question = Question(question=question_asked, answer=answer, context=context, audio_src_url="audio22")
    print(question.answer)
    
    return { "question": question.question, "answer": answer, "audio_src_url": question.audio_src_url, "id": question.pk }

if __name__ == "__main__":
    ques = "how to earn money by build a small company?"
    book_path = r"D:\Study\python style of Google.pdf.pages.csv"

    question_1 = "\n\n\nQ: How to choose what business to start?\n\nA: First off don't be in a rush. Look around you, see what problems you or other people are facing, and solve one of these problems if you see some overlap with your passions or skills. Or, even if you don't see an overlap, imagine how you would solve that problem anyway. Start super, super small."
    question_2 = "\n\n\nQ: Q: Should we start the business on the side first or should we put full effort right from the start?\n\nA:   Always on the side. Things start small and get bigger from there, and I don't know if I would ever “fully” commit to something unless I had some semblance of customer traction. Like with this product I'm working on now!"
    question_3 = "\n\n\nQ: Should we sell first than build or the other way around?\n\nA: I would recommend building first. Building will teach you a lot, and too many people use “sales” as an excuse to never learn essential skills like building. You can't sell a house you can't build!"
    question_4 = "\n\n\nQ: Andrew Chen has a book on this so maybe touché, but how should founders think about the cold start problem? Businesses are hard to start, and even harder to sustain but the latter is somewhat defined and structured, whereas the former is the vast unknown. Not sure if it's worthy, but this is something I have personally struggled with\n\nA: Hey, this is about my book, not his! I would solve the problem from a single player perspective first. For example, Gumroad is useful to a creator looking to sell something even if no one is currently using the platform. Usage helps, but it's not necessary."
    question_5 = "\n\n\nQ: What is one business that you think is ripe for a minimalist Entrepreneur innovation that isn't currently being pursued by your community?\n\nA: I would move to a place outside of a big city and watch how broken, slow, and non-automated most things are. And of course the big categories like housing, transportation, toys, healthcare, supply chain, food, and more, are constantly being upturned. Go to an industry conference and it's all they talk about! Any industry…"
    question_6 = "\n\n\nQ: How can you tell if your pricing is right? If you are leaving money on the table\n\nA: I would work backwards from the kind of success you want, how many customers you think you can reasonably get to within a few years, and then reverse engineer how much it should be priced to make that work."
    question_7 = "\n\n\nQ: Why is the name of your book 'the minimalist entrepreneur' \n\nA: I think more people should start businesses, and was hoping that making it feel more “minimal” would make it feel more achievable and lead more people to starting-the hardest step."
    question_8 = "\n\n\nQ: How long it takes to write TME\n\nA: About 500 hours over the course of a year or two, including book proposal and outline."
    question_9 = "\n\n\nQ: What is the best way to distribute surveys to test my product idea\n\nA: I use Google Forms and my email list / Twitter account. Works great and is 100% free."
    question_10 = "\n\n\nQ: How do you know, when to quit\n\nA: When I'm bored, no longer learning, not earning enough, getting physically unhealthy, etc… loads of reasons. I think the default should be to “quit” and work on something new. Few things are worth holding your attention for a long period of time."
    header = """Sahil Lavingia is the founder and CEO of Gumroad, and the author of the book The Minimalist Entrepreneur (also known as TME). These are questions and answers by him. Please keep your answers to three sentences maximum, and speak in complete sentences. Stop speaking once your point is made.\n\nContext that may be useful, pulled from The Minimalist Entrepreneur:\n"""

    exam_QA = ExampleQA(header=header,
                        QA_list=[question_1, question_2, question_3, question_4, question_5, question_6, question_7, question_8, question_9, question_10])
    ask({"question": ques}, book_path, exam_QA)
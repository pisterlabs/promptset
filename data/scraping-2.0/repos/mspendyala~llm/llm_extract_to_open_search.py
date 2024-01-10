# pip install fitz PyPDF2 PyMuPDF openai opensearch-py

import fitz
import openai
from opensearchpy import OpenSearch, RequestsHttpConnection
import uuid
import pprint

pp = pprint

openai.api_key = 'get your own key'


def extract_from_pdf(pdf_path):
    """
    Extract content from a given PDF.

    Args:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - List[Dict]: List containing dictionaries with text, images, and page number from each page.
    """
    doc = fitz.open(pdf_path)
    content_data = []

    for page_num, page in enumerate(doc):
        #images = page.getImageList()
        text = page.get_text()

        content_data.append({
            "text": text,
            "page_number": page_num
        })
        #
        # content_data.append({
        #     "text": text,
        #     "images": images,
        #     "page_number": page_num,
        # })

    return content_data


def generate_qa_pairs_from_text(text):
    """
    Generate question-answer pairs based on the given content using OpenAI's GPT model.

    Args:
    - text (str): Content text.

    Returns:
    - List[dict]: List of dictionaries with keys 'question' and 'answer'.
    """
    prompt = f"Given the text: \"{text}\", generate a set of questions and answers based on the content."
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        max_tokens=100
    )
    output = response.choices[0].text.strip().split("\n")

    qa_pairs = []
    for i in range(0, len(output) - 1, 2):  # loop in steps of 2
        if output[i].startswith("Question") and "Answer:" in output[i + 1]:
            question = output[i].split(": ")[1] if ": " in output[i] else output[i]
            answer = output[i + 1].split(": ")[1] if ": " in output[i + 1] else output[i + 1]
            qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs


# def generate_qa_from_text(text):
#     """
#     Generate questions and answers based on the given content using OpenAI's GPT model.
#
#     Args:
#     - text (str): Content text.
#
#     Returns:
#     - List[Dict]: Generated question and answer pairs.
#     """
#     prompt = f"Given the text: \"{text}\", generate a set of questions and their corresponding answers."
#     response = openai.Completion.create(
#         model='text-davinci-003',
#         prompt=prompt,
#         max_tokens=100
#     )
#     output = response.choices[0].text.strip().split("\n")
#     questions = []
#     for line in output:
#         if line.startswith("Question"):
#             parts = line.split(": ")
#             if len(parts) > 1:  # ensures there's something after the colon
#                 questions.append(parts[1])
#
#     return questions
#     #questions, answers = output.split("\n")[::2], output.split("\n")[1::2]
#     #questions = output.split("\n")[::2]
#     # questions = [q for q in output.split("\n")[::2] if len(q.strip()) > 1]
#     # print("questions")
#     # pp.pprint(questions)
#     # #return [{"question": q, "answer": a} for q, a in zip(questions, answers)]
#     # return [{"question": q} for q in zip(questions)]


def generate_embedding(question):
    """
    Generate embeddings for the provided question using OpenAI's embedding model.

    Args:
    - question (str): Generated question.

    Returns:
    - List[float]: Embedding vector.
    """
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=question
    )
    # Access the embedding correctly
    embedding = response['data'][0]['embedding']
    return embedding


def opensearch_connection():
    """
    Create a connection to the OpenSearch cluster.

    Returns:
    - OpenSearch: Client instance.
    """
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_auth=('admin', 'admin'),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False
    )
    return client

def store_in_opensearch(data, unique_id):
    """
    Store content data in OpenSearch.

    Args:
    - data (Dict): Content data.
    - unique_id (str): Unique identifier.
    """
    data["uuid"] = unique_id
    client = opensearch_connection()
    client.index(index="content_data_index_star", body=data)

def store_question_answer_vector(question, answer, embedding, unique_id):
    """
    Store question embeddings in OpenSearch.

    Args:
    - question (str): Generated question.
    - embedding (List[float]): Embedding vector.
    - unique_id (str): Unique identifier.
    """
    embedding_data = {
        "question": question,
        "answer": answer,
        "embedding": embedding,
        "uuid": unique_id
    }
    print("embedding data")
    pp.pprint(embedding_data)
    client = opensearch_connection()
    client.index(index="question_embedding_index_star", body=embedding_data)

def retrieve_data_from_opensearch(unique_id):
    """
    Retrieve content data from OpenSearch using a UUID.

    Args:
    - unique_id (str): Unique identifier.

    Returns:
    - Dict: Content data.
    """
    client = opensearch_connection()
    response = client.search(index="content_data_index_star", body={
        "query": {
            "match": {
                "uuid": unique_id
            }
        }
    })
    return response['hits']['hits'][0]['_source']

def main_workflow(pdf_path):
    """
    Primary workflow: Extracts data from PDF, generates Q&A, creates embeddings, and stores them in OpenSearch.

    Args:
    - pdf_path (str): Path to the PDF file.
    """
    extracted_data = extract_from_pdf(pdf_path)
    print(extracted_data)

    for data in extracted_data:
        print(data)
        qa_pairs = generate_qa_pairs_from_text(data['text'])
        print(qa_pairs)
        for qa_pair in qa_pairs:
            question = qa_pair['question']
            answer = qa_pair['answer']
            print("question")
            print(question)
            embedding = generate_embedding(question)
            print(embedding)
            unique_id = str(uuid.uuid4())
            print("uni")
            print(unique_id)
            store_in_opensearch(data, unique_id)
            qa_data = {

            }
            store_question_answer_vector(question, answer, embedding, unique_id)



main_workflow(pdf_path='pen.pdf')
print("retrieve_data")
pp.pprint(retrieve_data_from_opensearch(unique_id='4da40e2c-dbe1-4db9-b4ba-b99aaf8b6692'))

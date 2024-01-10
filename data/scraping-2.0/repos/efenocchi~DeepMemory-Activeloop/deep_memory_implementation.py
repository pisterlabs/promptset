# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, missing-function-docstring
import getpass
import os
import argparse


from deeplake import VectorStore
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores.deeplake import DeepLake
from global_variables import TYPE_BIOMEDICAL, TYPE_LEGAL, TYPE_FINANCE

# from upload_existing_dataset import upload_with_deepcopy

# from dataset_generator import get_chunk_qa_data
from dataset_generator_langchain import get_chunk_qa_data_old

from global_variables import YAML_FILE, HUB_NAME

TYPE = TYPE_FINANCE

parser = argparse.ArgumentParser()
parser.add_argument("--credentials", action="store_true")
args = parser.parse_args()

if args.credentials:
    os.environ["ACTIVELOOP_TOKEN"] = getpass.getpass(
        "Copy and paste your ActiveLoop token: "
    )
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Copy and paste your OpenAI API key: "
    )

else:
    os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings_function = OpenAIEmbeddings()


def load_vector_store(user_hub, name_db):
    vector_store_db = DeepLake(
        f"hub://{user_hub}/{name_db}",
        embedding_function=embeddings_function.embed_documents,
        runtime={"tensor_db": True},
        read_only=True,
    )
    return vector_store_db.vectorstore


def training_job(vector_store_db, chunk_question_quantity: int):
    # chunks, chunk_id = [str(el.text.data()['value']), str(el.id.data()['value']) for idx, el in enumerate(vector_store_db.dataset) if idx < first_chunks]  # first N chunks

    # create real cluster
    # embeddings = vector_store_db.dataset.embedding.data()["value"]
    # elbow_technique(embeddings)

    # create cluster

    questions = []
    relevances = []

    for idx, el in enumerate(vector_store_db.dataset):
        if idx >= chunk_question_quantity:
            break
        print(f"Generating question: {idx}")
        chunk_id = str(el.id.data()["value"])
        text = str(el.text.data()["value"])
        print(f"Processing chunk: {idx}")
        single_question, single_relevance = get_chunk_qa_data_old(text)
        questions.append(single_question)
        # relevances.append([(chunk_id, float(single_relevance))])
        relevances.append([(chunk_id, 1)])

    job_id = vector_store_db.deep_memory.train(
        queries=questions,
        relevance=relevances,
        embedding_function=embeddings_function.embed_documents,
    )
    vector_store_db.deep_memory.status(job_id)
    return vector_store_db


def get_answer(vector_store_db, user_question, deep_memory):
    # deep memory inside the vectore store ==> deep_memory=True
    answer = vector_store_db.search(
        embedding_data=user_question,
        embedding_function=embeddings_function.embed_query,
        deep_memory=deep_memory,
        return_view=False,
    )
    print(answer)

    return answer


def check_status(job_id: str, vectore_store_db: str):
    db = VectorStore(vectore_store_db, read_only=True)
    status = db.deep_memory.status(job_id)
    print(status)


def evaluate_deep_memory(vector_store_db, train_question_quantity, number_of_questions):
    test_questions = []
    test_relevance = []

    dataset_unseen = vector_store_db.dataset[train_question_quantity:]
    for idx, el in enumerate(dataset_unseen):
        if idx >= number_of_questions:
            break
        print(f"Generating question: {idx}")
        chunk_id = str(el.id.data()["value"])
        text = str(el.text.data()["value"])
        single_question, single_relevance = get_chunk_qa_data_old(text)
        test_questions.append(single_question)

        test_relevance.append([(chunk_id, 1)])
    evaluation = vector_store_db.deep_memory.evaluate(
        queries=test_questions,
        relevance=test_relevance,
        embedding_function=embeddings_function.embed_documents,
        top_k=[1, 3, 5, 10, 50, 100],
    )
    with open(f"question_evaluation_{YAML_FILE['db'][TYPE]['name']}.txt", "w") as file:
        file.write(f"Test Question: {test_questions}\n")
        file.write(f"Evaluation Result: {evaluation}\n")
    return evaluation


if __name__ == "__main__":
    DATASET_NAME = YAML_FILE["db"][TYPE]["name"]
    train_question_quantity = YAML_FILE["db"][TYPE]["query_numbers"]

    vector_store_db = load_vector_store(HUB_NAME, DATASET_NAME)

    # UPLOAD THE DATASET WITH DEEPCOPY
    # upload_with_deepcopy("embedding_custom4_11_13_2023")

    # TRAIN PHASE
    # training_job(vector_store_db, train_question_quantity)
    # print(vector_store_db.deep_memory.list_jobs())
    # user_question = "Female carriers of the Apolipoprotein E4 (APOE4) allele have increased risk for dementia."

    # TEST PHASE
    # search_response_deep_memory = get_answer(vector_store_db, user_question, deep_memory=True)
    # search_response_standard = get_answer(vector_store_db, user_question, deep_memory=False)

    evaluation = evaluate_deep_memory(
        vector_store_db, train_question_quantity, number_of_questions=100
    )
    # job_id = YAML_FILE["db"][TYPE]["job_id"]
    # vectore_store_db = f"""hub://manufe/{YAML_FILE["db"][TYPE]["name"]}"""
    # check_status(
    #     job_id=job_id,
    #     vectore_store_db=vectore_store_db,
    # )
    print(evaluation)

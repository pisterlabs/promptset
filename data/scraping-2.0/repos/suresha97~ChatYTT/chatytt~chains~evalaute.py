from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

from chatytt.chains.standard import ConversationalQAChain
from chatytt.chains.custom import (
    ConversationalQASequentialChain,
    ConversationalQALCELChain,
)
from chatytt.vector_store.pinecone_db import PineconeDB


def get_example_docs(vector_store, sample_size):
    # Hack since it's not currently possible to retrieve a complete set of ids in a Pinecone Index
    # https://community.pinecone.io/t/returning-list-of-ids/140
    docs = vector_store.similarity_search(query="", k=sample_size)

    return [doc.page_content for doc in docs]


def generate_examples_qas(docs):
    example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())
    examples = example_gen_chain.apply_and_parse([{"doc": t} for t in docs])

    return [example["qa_pairs"] for example in examples]


def get_chain_predictions(chain, example_qas):
    predictions = []

    for example in example_qas:
        response = chain.get_response(query=example["query"], chat_history=[])
        predictions.append({"result": response})

    return predictions


def evaluate_predictions_against_examples_qas(examples, predictions, verbose=False):
    llm = ChatOpenAI(temperature=0)
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)

    if verbose:
        for i, eg in enumerate(examples):
            print(f"Example {i}:")
            print("Question: " + examples[i]["query"])
            print("Real Answer: " + examples[i]["answer"])
            print("Predicted Answer: " + predictions[i]["result"])
            print("Predicted Grade: " + graded_outputs[i]["results"])
            print()

    total_correct = sum(grade["results"] == "CORRECT" for grade in graded_outputs)

    return total_correct


if __name__ == "__main__":
    load_dotenv()
    pinecone_db = PineconeDB(
        index_name="youtube-transcripts", embedding_source="open-ai"
    )
    vector_store = pinecone_db.vector_store

    example_docs = get_example_docs(vector_store, sample_size=10)
    example_qas = generate_examples_qas(example_docs)

    chains = {
        "standard_qa": ConversationalQAChain(vector_store=vector_store),
        "custom_lcel_qa": ConversationalQALCELChain(
            vector_store=pinecone_db.vector_store,
            chat_model=ChatOpenAI(temperature=0.0),
        ),
        "custom_seq_qa_chain": ConversationalQASequentialChain(
            vector_store=pinecone_db.vector_store,
            chat_model=ChatOpenAI(temperature=0.0),
        ),
    }

    for chain_name, chain in chains.items():
        predictions = get_chain_predictions(chain, example_qas)

        total_correct = evaluate_predictions_against_examples_qas(
            example_qas, predictions, verbose=False
        )

        print(
            f"{chain_name} chain got {total_correct} out of {len(example_qas)} correct!"
        )

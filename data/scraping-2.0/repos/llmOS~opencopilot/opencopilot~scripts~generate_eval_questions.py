import argparse
import json
from dataclasses import dataclass
from typing import List
import warnings

from tqdm import tqdm
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
import settings
from src.repository.documents import document_store

from src.eval.entities import RetrievalExample, RetrievalDataset, DocumentID

PROMPT_TEMPLATE = """"
You are a professional game developer with 20 years of experience.
Your goal is to help other developers integrate Ready Player Me Avatar SDK.

Please generate 5 questions from given url and context which developers might ask. Please think this through step by step so we come to factual and helpful questions.
Avoid generating bad questions.

GOOD QUESTIONS:
What do I need to do to use Ready Player Me avatars for commercial use?
Can I control the logging feature of the avatar loading process?
What should I do to avoid errors when uploading to Mixamo?
How can I add animations to Ready Player Me avatars?
How can I optimize the avatars for my game?

BAD QUESTIONS:
What does the WebViewActivity.kt file do?
What is the 'GetAvatarCacheData' function used for?
What is the purpose of the AvatarRenderLoader class?
What is the ReadyPlayerMeComponent?
What is the purpose of the 'FDownloadImageCompleted& OnCompleted' parameter in the 'Load' function?

Output questions in JSON list format like [{{"query": "First question?", "answer": "First answer"}},...].

URL: {url}
CONTEXT: {context}
"""

PROMPT = PromptTemplate(input_variables=["url", "context"], template=PROMPT_TEMPLATE)


@dataclass(frozen=True)
class DocumentQueries:
    source_document: DocumentID
    queries: List[str]


def main(dataset_path: str) -> None:
    print("question generation starting!")
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")
    # TODO: document_store.init_document_store()
    store = document_store.get_document_store()
    documents = store.load_documents()
    generated_questions = []
    for document in tqdm(documents):
        response = llm(
            [
                SystemMessage(
                    content=PROMPT_TEMPLATE.format(
                        url=document.metadata["source"], context=document.page_content
                    )
                )
            ]
        )
        if questions := _parse_questions(response.content, document.metadata["source"]):
            generated_questions.append(questions)
    _save_dataset(generated_questions, dataset_path)


def _save_dataset(doc_queries_list: List[DocumentQueries], dataset_path: str):
    examples = []
    for doc_queries in doc_queries_list:
        for query in doc_queries.queries:
            retrieval_example = RetrievalExample(
                query=query, documents=[doc_queries.source_document]
            )

            examples.append(retrieval_example)

    dataset = RetrievalDataset(examples=examples)

    # Writing to JSON
    with open(dataset_path, "w") as file:
        json.dump(dataset.to_dict(), file, indent=4)


def _parse_questions(
    llm_response: str, source_document: DocumentID
) -> List[DocumentQueries]:
    queries = []
    try:
        json_llm_response = json.loads(llm_response)
        for response in json_llm_response:
            queries.append(response["query"])
    except:
        warnings.warn(
            f"Could not parse LLM response for document {source_document}: {llm_response}"
        )
    return DocumentQueries(queries=queries, source_document=source_document)


if __name__ == "__main__":
    assert settings.get().OPENAI_API_KEY, "OpenAI API key needs to be present"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="../copilots/rpm/eval_data/retrieval_auto.json",
    )
    args = parser.parse_args()
    main(args.dataset_path)

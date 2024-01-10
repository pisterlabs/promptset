# from pathlib import Path
# from pprint import pprint
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLM
from ragas.testset import TestsetGenerator

from regent_rag.core.logging import logger


def main() -> None:
    # Define the metadata extraction function.
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["source"] = record.get("source")
        return metadata

    loader = JSONLoader(
        file_path="./out/train.jsonl",
        jq_schema=".",
        content_key="text",
        metadata_func=metadata_func,
        json_lines=True,
    )

    logger.info("Loading documents...")
    documents = loader.load()

    # pprint(documents)

    # testsetgenerator = TestsetGenerator.from_default()
    # test_size = 1
    # testset = testsetgenerator.generate(documents, test_size=test_size)

    # Add custom llms and embeddings
    generator_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-3.5-turbo"))
    critic_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-4"))
    embeddings_model = OpenAIEmbeddings()

    # Change resulting question type distribution
    testset_distribution = {
        "simple": 0.25,
        "reasoning": 0.5,
        "multi_context": 0.0,
        "conditional": 0.25,
    }

    # percentage of conversational question
    chat_qa = 0.2

    test_generator = TestsetGenerator(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings_model=embeddings_model,
        testset_distribution=testset_distribution,
        chat_qa=chat_qa,
    )

    logger.info("Generating tests...")
    testset = test_generator.generate(documents, test_size=50)

    logger.info("Generating DataFrame...")
    test_df = testset.to_pandas()
    test_df.head()
    test_df.to_json("./out/dataset.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()

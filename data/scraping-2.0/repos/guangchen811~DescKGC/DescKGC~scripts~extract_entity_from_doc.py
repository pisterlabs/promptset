import argparse

from langchain.chat_models import ChatOpenAI

from DescKGC.procedures.extract_and_insert.base import (
    extract_entities_from_paper, get_paper_title_by_field)
from DescKGC.procedures.build_config import load_config
from DescKGC.tools.db_manager import DBManager
from DescKGC.tools.extractor.base import init_extract_chain


def add_arguments(parser):
    parser.add_argument("--temperature", type=float)


def main(args):
    config = load_config()
    if args.temperature is None:
        temperature = config["llm"]["temperature"]
    else:
        temperature = args.temperature
    llm = ChatOpenAI(temperature=temperature)
    extract_chain = init_extract_chain(llm)
    db_manager = DBManager(**config["neo4jdb"], **config["chromadb"])

    res = get_paper_title_by_field(db_manager, "doi")
    for title in res:
        try:
            extract_entities_from_paper(
                paper_id_type="title",
                paper_id_value=title,
                topic=config["topic"],
                shortenings=config["shortenings"],
                vs_key_info=config["extractor"]["entity"]["vs_key_info"],
                db_manager=db_manager,
                extract_chain=extract_chain,
            )
        except Exception as e:
            print(f"Error when extracting entities from {title}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

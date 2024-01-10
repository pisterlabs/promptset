from langchain.chains import AnalyzeDocumentChain
from langchain.chat_models.gigachat import GigaChat
from langchain.chains.question_answering import load_qa_chain
from config_data.config import Config, load_config
from prompt_stats.summarize_stats import get_summarize_stats


@get_summarize_stats
def get_long_answer(file_path: str, question: str):
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    with open(file_path, encoding='utf-8') as f:
        state_of_the_union = f.read()

    qa_chain = load_qa_chain(giga, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

    return qa_document_chain.run(
        input_document=state_of_the_union,
        question=question,
    )

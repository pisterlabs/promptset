from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from configs.logger import print_info_log, catch_error
from constants.log_tags import LogTag

CONTEXT_TOKENS_COUNT = 4000

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CONTEXT_TOKENS_COUNT,
    chunk_overlap=0,
    separators=["."]
)


def split_text_to_chunks(text: str, project_id: str, show_logs: bool) -> List[str]:
    """
    Splits a given text into chunks.

    :param text: The text to split.
    :param project_id: The id of the processing project.
    :param show_logs: Determines whether to display logs while splitting text.

    :return: The list of string text chunks.
    """

    try:
        if show_logs:
            print_info_log(
                tag=LogTag.SPLIT_TEXT_TO_CHUNKS,
                message=f"Splitting text by {CONTEXT_TOKENS_COUNT} tokens..."
            )

        text_chunks = text_splitter.split_text(text)

        if show_logs:
            print_info_log(
                tag=LogTag.SPLIT_TEXT_TO_CHUNKS,
                message=f"Text splitting completed."
            )

        return text_chunks

    except Exception as e:
        catch_error(
            tag=LogTag.SPLIT_TEXT_TO_CHUNKS,
            error=e,
            project_id=project_id
        )

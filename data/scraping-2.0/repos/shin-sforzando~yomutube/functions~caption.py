import re

from langchain.document_loaders import YoutubeLoader
from models import Video
from utils import NO_CAPTION_MESSAGE


def get_formatted_text(text: str) -> str:
    """
    Removes square brackets and their contents from the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The formatted text with square brackets removed.
    """
    text = re.sub(r"\[.*\]", "", text)
    return text


def get_caption(
    video: Video, preferred_language: str = "ja", lower_limit_chars: int = 256
) -> str:
    """
    Retrieves the caption for a given video.

    Args:
        video (Video): The video object for which to retrieve the caption.
        preferred_language (str, optional): The preferred language for the caption. Defaults to "ja".
        lower_limit_chars (int, optional): The lower limit of characters for the caption. Defaults to 64.

    Returns:
        str: The formatted caption text.

    Raises:
        Exception: If an error occurs during the caption retrieval process.
    """
    try:
        loader = YoutubeLoader(
            video_id=video.id,
            add_video_info=False,
            language=["ja", "en"],
            translation=preferred_language,
            continue_on_failure=True,
        )
        documents = loader.load()
        if 0 < len(documents):
            formatted_text = get_formatted_text(documents[0].page_content)
            if lower_limit_chars < len(formatted_text):
                return formatted_text
            else:
                print(f'"{formatted_text}" ({len(formatted_text)} chars) is too short.')
    except Exception as e:
        print(e)
    print(f"Failed to retrieve caption for {video.id}.")
    return NO_CAPTION_MESSAGE

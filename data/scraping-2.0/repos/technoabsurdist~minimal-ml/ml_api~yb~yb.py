from langchain.document_loaders import YoutubeLoader
import sieve
import shutil

def yb_transcript(url: str):
    """
    Retrieves the transcript of a YouTube video given its URL.

    Uses the 'YoutubeLoader' from 'langchain.document_loaders' to fetch the transcript of a YouTube video.
    The transcript is returned as a list, where each element represents a segment of the video's content.

    Parameters:
    - url (str): The URL of the YouTube video for which the transcript is required.

    Returns:
    - list: A list of strings, each representing a segment of the video's transcript.

    Note:
    - The function does not include video information in the transcript.
    - The returned list contains the textual content as it appears in the video's closed captions.
    """
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    result = loader.load()
    tables = [doc for doc in result]
    return [table.page_content for table in tables]


def yb_download(url: str):
    """
    Downloads a YouTube video as an MP4 file given its URL.

    Utilizes the 'sieve/youtube_to_mp4' function to download a YouTube video. 
    The video is saved to a specified destination path in MP4 format.

    Parameters:
    - url (str): The URL of the YouTube video to be downloaded.

    Returns:
    - str: The file path to the downloaded MP4 file.

    Note:
    - The function saves the downloaded video in the 'downloads' directory with the name 'yb_download.mp4'.
    """
    youtube_to_mp4 = sieve.function.get("sieve/youtube_to_mp4")
    output = youtube_to_mp4.run(url, True)
    destination_path = "downloads/yb_download.mp4"
    shutil.copy(output.path, destination_path)

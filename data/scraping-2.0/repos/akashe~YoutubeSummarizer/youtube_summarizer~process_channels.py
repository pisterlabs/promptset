import asyncio

import argparse
from typing import List
import openai

from youtube.get_information import YoutubeConnect

from utils import get_adjusted_iso_date_time, check_supported_models, get_transcripts, http_connection_decorator
from get_chain import aget_summary_of_each_video, get_documents, aget_summary_with_keywords

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from openai_prompts import get_per_document_with_keyword_prompt_template, \
    get_combine_document_prompt_template, \
    get_per_document_prompt_template, \
    get_combine_document_with_source_prompt_template


#["https://www.youtube.com/@BeerBiceps", "https://www.youtube.com/@hubermanlab","https://www.youtube.com/@MachineLearningStreetTalk"]
#["AGI", "history", "spirituality", "human pyschology", "new developments in science"]
async def process_channels(
    youtube_channel_links: List[str] = ["https://www.youtube.com/@BeerBiceps", "https://www.youtube.com/@hubermanlab","https://www.youtube.com/@MachineLearningStreetTalk"],
    summary_of_n_weeks: int = 1,
    search_terms: List[str] = None,
    get_source: bool = False,
    model_name: str = "gpt-4-1106-preview"
) -> str:

    latest_video_ids = []
    youtube_connect = YoutubeConnect()

    for youtube_channel_link in youtube_channel_links:
        #TODO: add method using undername
        channel_id = youtube_connect.get_channel_id_from_channel_link(youtube_channel_link)
        logger.info(f"Found channel id {channel_id} for the youtube channel at {youtube_channel_link}")

        publish_date_after = get_adjusted_iso_date_time(summary_of_n_weeks)
        video_ids = youtube_connect.get_latest_videos(channel_id, publish_date_after)

        if video_ids is not None:
            latest_video_ids.extend(video_ids)

    logger.info(f"Analyzing a total of {len(latest_video_ids)} videos")
    print("\n")
    if len(latest_video_ids) == 0:
        print(f"No videos uploaded in the last {summary_of_n_weeks} week")
        return "-1"
    elif len(latest_video_ids) > 1:
        print(f"Analyzing a total of {len(latest_video_ids)} videos")
    else:
        print(f"Analyzing a total of {len(latest_video_ids)} video")

    print("\n")
    video_titles = []
    for id, video_id in enumerate(latest_video_ids):
        video_title = youtube_connect.get_video_title(video_id)
        video_titles.append(video_title)
        print(f"{id+1}. [{video_title}](https://www.youtube.com/watch?v={video_id})")

    assert len(video_titles) == len(latest_video_ids)

    transcripts = get_transcripts(latest_video_ids, video_titles)

    documents = get_documents(latest_video_ids, video_titles, transcripts, model_name)

    result = ""
    try:
        if search_terms:
            per_document_template = get_per_document_with_keyword_prompt_template(model_name)
            combine_document_template = get_combine_document_with_source_prompt_template(model_name) if get_source \
                else get_combine_document_prompt_template(model_name)

            result = await aget_summary_with_keywords(documents,
                                                      search_terms,
                                                      per_document_template,
                                                      combine_document_template,
                                                      model_name,
                                                      len(latest_video_ids))
        else:
            per_document_template = get_per_document_prompt_template(model_name)
            result = await aget_summary_of_each_video(documents, per_document_template, model_name)

    except Exception as e:
        print("Something bad happened with the request. Please retry :)")
        return "-1"

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('youtube_channel_links', nargs='+', metavar='y',
                        help='a list youtube channel links that you want to process')
    parser.add_argument('-n', '--summary_of_n_weeks', type=int, default=1,
                        help='This would be used to collect videos released in past n weeks')
    parser.add_argument('-s', '--search_terms', type=str, nargs='*',
                        help="design the summary around your topics of interest. If not given,"
                             "a general summary will be created.")
    parser.add_argument('--return_sources', action='store_true', default=False,
                        help="To return sources of information in the final summary.")
    parser.add_argument('--model_name', default='gpt-3.5-turbo-16k',
                        help="model to use for generating summaries.")

    args = parser.parse_args()

    assert check_supported_models(args.model_name), "Model not available in config"

    asyncio.run(
        process_channels(args.youtube_channel_links, args.summary_of_n_weeks, args.search_terms, args.return_sources)
    )


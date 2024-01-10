from langchain.docstore.document import Document
from fastapi import APIRouter
import functools
import redis

from typing import List

from .core import check_cache, save_to_cache, upload_summary_to_s3, get_transcript_from_s3, SummaryRequestModel, \
    generate_summary, get_summary_from_s3, create_html_bullet_point
from app.core.config import settings
from loguru import logger
import orjson as json

router = APIRouter()

@functools.lru_cache()
def get_redis_instance(): 
    redis_client = redis.Redis(host=settings.SUMM_REDIS_HOST, 
                               port=settings.SUMM_REDIS_PORT, 
                               password=settings.SUMM_REDIS_PASSWD)
    return redis_client

REDIS_INSTANCE = get_redis_instance()

@router.get("/fetchSummary")
async def fetchSummary(courseName: str, videoName: str) -> dict:
    """Given a course name and video name check the cache to see if we have a summary already generated

    Args:
        courseName (str): Name of course the video comes from
        videoName (str): The video for which the transcript we want to summarize

    Returns:
        dict: Summary results or None
    """
    if videoName:
        cache_results: dict | None = check_cache(course_name=courseName, video_name=videoName, 
                                             redis_instance=REDIS_INSTANCE)
    if cache_results:
        logger.info("Found in cache")
        cache_results['summary'] = create_html_bullet_point(cache_results['summary'].decode('utf-8'))
        cache_results.update({"status": True, "msg": "Success"})
        return cache_results
    else:
        logger.info("Not in cache, checking s3")
        db_results: str | None = get_summary_from_s3(course_name=courseName, video_name=videoName)
        if db_results:
            db_results['summary'] = create_html_bullet_point(db_results['summary'])
            db_results.update({"status": True, "msg": "Success"})
            return db_results
        return {"status": False, "msg": "Not Found"}

@router.post("/generateSummary")
async def generateSummary(summary_model: SummaryRequestModel) -> dict:
    """ 
    Given an s3 path to a video transcript, load the transcript, summarize it using the llm, save the results to the 
    cache and database, and return the results.

    Args:
        summary_model (SummaryRequestModel): Input data from the http request

    Returns:
        dict: The summary of the transcript and some meta data or an error that the summary could not be generated
    """
    failure_dict =  {
        "msg": f"Could not generate summary for video: {summary_model.videoName} in course: " 
        f"{summary_model.courseName}",
        "status": False
    }
    if summary_model.s3_path:
        transcripts_to_summarize: List[Document] | None = get_transcript_from_s3(s3_path=summary_model.s3_path,)
        if transcripts_to_summarize:
            txt = transcripts_to_summarize[0]
        else:
            return failure_dict
    elif not summary_model.transcript:
        return failure_dict
    else:
        txt = Document(page_content=summary_model.transcript)
    try:
        summary_result = generate_summary(txt_to_summarize=txt, 
                                        gpt_model_name=settings.GPT_MODEL_NAME) # Assuming only one doc
        save_to_cache(course_name=summary_model.courseName, video_name=summary_model.videoName, summary=summary_result, 
                    redis_instance=REDIS_INSTANCE)
        await upload_summary_to_s3(course_name=summary_model.courseName, transcript_name=summary_model.videoName,
                            summary_text=summary_result)
        result = {
            "summary": create_html_bullet_point(summary_result),
            "status": True,
            "msg": "success"
            }
        summary_dict = json.loads(summary_model.json())
        summary_dict.pop("transcript", "")
        result.update(summary_dict)
        return result
    except Exception as exception:
        logger.exception(exception)
        return failure_dict
    

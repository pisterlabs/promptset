from langchain.document_loaders import S3FileLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
from typing import Optional
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from redis import Redis
import aioboto3
import botocore
import tiktoken

from typing import List
import re 

from app.core.config import settings

class SummaryRequestModel(BaseModel):
    userId: int
    courseName: str
    videoName: str
    transcript: Optional[str]
    s3_path: Optional[str]


async def upload_summary_to_s3(course_name: str, transcript_name: str, summary_text: str):
    """Writes the generated summary to a database

    Args:
        course_name (str): course video is associated with
        transcript_name (str): video's transcript that's being summarized
        summary_text (str): Summary of transcript
    """

    session = aioboto3.Session()
    client = session.client(
        's3', 
        config=botocore.config.Config(s3={'addressing_style': 'virtual'}), 
        region_name=settings.AWS_REGION_NAME, 
        endpoint_url=settings.S3_ENDPOINT_URL, 
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
    )
    async with client as s3:
    # Upload the transcript text
        await s3.put_object(
            Bucket=settings.S3_BUCKET_NAME, 
            Key=f"{course_name}/summaries/{transcript_name}", 
            Body=summary_text, 
            ACL='private', 
            Metadata={
                'x-amz-meta-my-key': 'placeholder-value'
            }
        )

def get_transcript_from_s3(s3_path: str) -> List[Document]:
    """Retrieves the text transcript of a video from s3

    Args:
        s3_path (str): Amazon s3 path to access transcript 

    Returns:
        List[Document]: List of one Document object that is the transcript for a specific video
    """
    parsed_s3 = urlparse(url=s3_path, allow_fragments=False)
    loader = S3FileLoader(bucket=parsed_s3.netloc, key= parsed_s3.path.lstrip('/'), 
                          region_name=settings.AWS_REGION_NAME, endpoint_url=settings.S3_ENDPOINT_URL,
                          aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
    try:
        return loader.load()
    except ClientError:
        # Handles if the s3 path DNE
        return None

def get_summary_from_s3(course_name: str, video_name: str) -> dict | None:
    loader = S3FileLoader(bucket=settings.S3_BUCKET_NAME, key=f"{course_name}/summaries/{video_name}", 
                          region_name=settings.AWS_REGION_NAME, endpoint_url=settings.S3_ENDPOINT_URL,
                          aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
    try:
        doc: Document = loader.load()[0]
        return {
            "courseName": course_name, 
            "videoName": video_name, 
            "summary": doc.page_content
            }
    except ClientError:
        # Handles if the s3 path DNE
        return None

def check_cache(course_name: str, video_name: str, redis_instance: Redis) -> dict:
    """_summary_

    Args:
        course_name (str): course video is associated with
        video_name (str): video's transcript that's being summarized
        redis_instance (Redis): redis instance for caching
    Returns:
        dict: Returns summary and metadata 
    """
    cache_key = f"{course_name}/{video_name}"
    if redis_instance.exists(cache_key):
        return {
            "courseName": course_name, 
            "videoName": video_name, 
            "summary": redis_instance.hget(name=cache_key, key="summary")
            }
    else:
        return None

def save_to_cache(course_name: str, video_name: str, summary: str, redis_instance: Redis) -> None:
     """Saves generated summary to a cache for easy access

     Args:
        course_name (str): course video is associated with
        video_name (str): video's transcript that's being summarized
        summary (str): Summary of transcript
        redis_instance (Redis): redis instance for caching
     """
     redis_instance.hset(name=f"{course_name}/{video_name}",key="summary", value=summary)

def get_encoder_for_model(model_name: str): 
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding 

def get_text_splitter2(model_name, chunk_size=500, chunk_overlap=20, **kwargs):
    encoding = get_encoder_for_model(model_name).name
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding, keep_separator=False,
                                                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                add_start_index = True,  **kwargs)
    return text_splitter

def chunk_docs(docs,text_splitter, clean=True):
    if not isinstance(docs, list): 
        docs = [docs]           
    chunks = text_splitter.split_documents(docs)
    return chunks

def create_html_bullet_point(input_str: str, split_char="\n") -> str:
    """Converts a string of multiple items into an html bullet point list 

    Args:
        input_str (str): Summary string with each bullet point separated by /n/n

    Returns:
        str: html structure as a string e.g. <ul><li>bullet one</li></ul>
    """
    html_string = "<ul>"
    for bullet_point in input_str.split(split_char):
        if not bullet_point: continue 
        bullet_point = bullet_point.replace('-','', 1)
        html_string += f"<li>{bullet_point}</li>"
    html_string += "</ul>"
    return html_string

def generate_summary(txt_to_summarize: Document, gpt_model_name: str) -> str:
    """Uses an LLM to generate a summary for a transcript

    Args:
        txt_to_summarize (Document): The transcript to summarize

    Returns:
        str: Returns summarization of the transcript
    """
    llm = ChatOpenAI(temperature=0, model_name=gpt_model_name, openai_api_key=settings.OPENAI_API_KEY)

    # Remove spaces and new lines from text to summarize
    cleaned_txt = re.sub(r'[ |\t]+', ' ', txt_to_summarize.page_content)
    cleaned_txt = re.sub(r"\n+", "\n", cleaned_txt)
    txt_to_summarize.page_content = cleaned_txt

    chunk_summary_template = """
        Summarize this chunk of text that includes the main points and important details. {text}
    """
    prompt_token_amount = llm.get_num_tokens(chunk_summary_template)
    chunk_size = 4096 - prompt_token_amount # Ensures chunk + prompt < 4096 which is max chunk size

    # Chunk data
    text_splitter = get_text_splitter2(gpt_model_name, chunk_size=chunk_size, chunk_overlap=0)
    chunks = chunk_docs(txt_to_summarize, text_splitter=text_splitter)

    # Create prompts and templates
    chunk_summary_prompt = PromptTemplate(
        template=chunk_summary_template, 
        input_variables=["text"]
    )

    combine_summary_template="""
        Compose a concise and a brief summary of the following text: TEXT: `{text}`. Give the summary in bullet 
        points using a max of {num_bullet_points} concise bullet points.
    """

    combine_summary_prompt = PromptTemplate(
        input_variables=['text', 'num_bullet_points'],
        template=combine_summary_template
    )

    # Create map reduce summ chain and summarizes
    map_reduce_chain = load_summarize_chain(
        llm=llm, 
        map_prompt=chunk_summary_prompt, 
        combine_prompt=combine_summary_prompt, 
        chain_type="map_reduce", 
        return_intermediate_steps=False
    )
    num_of_tokens = llm.get_num_tokens(txt_to_summarize.page_content)
    num_of_bullet_points = -(num_of_tokens // -200) # Equivalent of math.ceil()
    summary = map_reduce_chain.run(**{'input_documents': chunks, 'num_bullet_points': num_of_bullet_points})
    return summary

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    import aiohttp
    import pinecone
    import tiktoken
    from langchain.document_loaders import BSHTMLLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from uuid import uuid4
    from tqdm.auto import tqdm
    import openai
    import os

def _get_delay_secs() -> float:
    return 3 

def tiktoken_len(text) -> int:
    tokenizer = tiktoken.get_encoding('p50k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def _get_local_path() -> Path:
    return Path(__file__).parent / "demo_fs"


def write_file(path: Path, body: str) -> None:
    """Write to the filesystem"""
    with open(path, "w") as handle:
        handle.write(body)


def read_file(path, url) -> list:
    """Read file and load with BS4"""
    loader = BSHTMLLoader(path)
    data = loader.load()
    plain_text = []
    for i in range(len(data)):
        a = data[i]
        plain_text.append({"text": a.page_content,
                            "source": url})
    return plain_text


def delete_file(path) -> None:
    """Convenience delete wrapper"""
    Path(path).unlink()


def create_filepath(unique_worker_id: str, workflow_uuid: str) -> Path:
    """Creates required folders and builds filepath"""
    directory = _get_local_path() / unique_worker_id
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / workflow_uuid
    return filepath


def process_file_contents(file_content: list) -> str:
    """split, create embeddings, and post to pinecone"""
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    
    for idx, record in enumerate(tqdm(file_content)):
        texts = text_splitter.split_text(record['text'])
        chunks.extend([{
          'id': str(uuid4()),
         'text': texts[i],
         'chunk': i,
          'url': record['source']
       } for i in range(len(texts))])
    
    openai.api_key = os.environ['OPENAI_API_KEY']
    embed_model = "text-embedding-ada-002"
    index_name = os.environ['PINECONE_INDEX']
    pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # app.pinecone.io (console)
    environment=os.environ['PINECONE_ENVIRONMENT']  # next to API key in console
    )
    
    res = openai.Embedding.create(
        input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
        ], engine=embed_model
    )
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=len(res['data'][0]['embedding']),
            metric='dotproduct'
        )
    # connect to index
    index = pinecone.GRPCIndex(index_name)

    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(chunks), batch_size)):
        # find end of batch
        i_end = min(len(chunks), i+batch_size)
        meta_batch = chunks[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        
        res = openai.Embedding.create(input=texts, engine=embed_model)
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{
            'text': x['text'],
            'chunk': x['chunk'],
            'url': x['url']
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)

    return f"Processed {len(chunks)} documents to pinecone"


@dataclass
class DownloadObj:
    url: str
    unique_worker_id: str
    workflow_uuid: str

@dataclass
class DownloadedObj:
    url: str
    path: str

@activity.defn
async def get_available_task_queue() -> str:
    """Just a stub for typedworkflow invocation."""
    raise NotImplementedError


@activity.defn
async def download_file_to_worker_filesystem(details: DownloadObj) -> str:
    """Download a URL to local filesystem"""
    # FS ops
    path = create_filepath(details.unique_worker_id, details.workflow_uuid)
    activity.logger.info(f"Downloading ${details.url} and saving to ${path}")

    # Here is where the real download code goes. Developers should be careful
    # not to block an async activity. If there are concerns about blocking download
    # or disk IO, developers should use loop.run_in_executor or change this activity
    # to be synchronous. Also like for all non-immediate activities, be sure to
    # heartbeat during download.
    async with aiohttp.ClientSession() as sess:
        async with sess.get(details.url) as resp:
            # We don't want to retry client failure
            if resp.status >= 400 and resp.status < 500:
                raise ApplicationError(f"Status: {resp.status}", resp.json(), non_retryable=True)
            # Otherwise, fail on bad status which will be inherently retried
            with open(path, 'wb') as fd:
                async for chunk in resp.content.iter_chunked(10):
                    fd.write(chunk)
    return str(path)


@activity.defn
async def work_on_file_in_worker_filesystem(dl_file: DownloadedObj) -> str:
    """Processing the file, in this case identical MD5 hashes"""
    content = read_file(dl_file.path, dl_file.url)
    checksum = process_file_contents(content)
    activity.logger.info(f"Did some work on {dl_file.path} with the URL {dl_file.url}, checksum {checksum}")
    return checksum


@activity.defn
async def clean_up_file_from_worker_filesystem(path: str) -> None:
    """Deletes the file created in the first activity, but leaves the folder"""
    activity.logger.info(f"Removing {path}")
    delete_file(path)


@workflow.defn
class FileProcessing:
    @workflow.run
    async def run(self, url: str) -> str:
        """Workflow implementing the basic file processing example.

        First, a worker is selected randomly. This is the "sticky worker" on which
        the workflow runs. This consists of a file download and the Pinecone pipeline,
        with a file cleanup if an error occurs.
        """
        workflow.logger.info("Searching for available worker")
        workflow.logger.info(f"url: {url}")
        unique_worker_task_queue = await workflow.execute_activity(
            activity=get_available_task_queue,
            start_to_close_timeout=timedelta(seconds=120),
        )
        workflow.logger.info(f"Matching workflow to worker {unique_worker_task_queue}")

        download_params = DownloadObj(
            url=url,
            unique_worker_id=unique_worker_task_queue,
            workflow_uuid=str(workflow.uuid4()),
        )

        download_path = await workflow.execute_activity(
            download_file_to_worker_filesystem,
            download_params,
            start_to_close_timeout=timedelta(seconds=120),
            task_queue=unique_worker_task_queue,
        )

        downloaded_file = DownloadedObj(url=url, path=download_path)

        checksum = "failed execution"  # Sentinel value
        try:
            checksum = await workflow.execute_activity(
                work_on_file_in_worker_filesystem,
                downloaded_file,
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(
                    maximum_attempts=2,
                    # maximum_interval=timedelta(milliseconds=500),
                ),
                task_queue=unique_worker_task_queue,
            )
        finally:
            await workflow.execute_activity(
                clean_up_file_from_worker_filesystem,
                download_path,
                start_to_close_timeout=timedelta(seconds=120),
                task_queue=unique_worker_task_queue,
            )
        return checksum
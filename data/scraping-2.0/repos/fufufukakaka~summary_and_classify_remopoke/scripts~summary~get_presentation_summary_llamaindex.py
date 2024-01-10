import dataclasses
import os

import click
import openai
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import (
    LangchainEmbedding,
    ServiceContext,
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llms import OpenAI
from tqdm.auto import tqdm
from youtube_transcript_api._errors import TranscriptsDisabled

openai.api_key = os.environ["OPENAI_API_KEY"]


@dataclasses.dataclass
class VideoSummary:
    title: str
    url: str
    summary: str


def get_video_summary(loader: YoutubeTranscriptReader, video_url: str) -> str:
    try:
        documents = loader.load_data(ytlinks=[video_url], languages=["ja"])
    except TranscriptsDisabled:
        return ""
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query(
        "この動画はポケモンに関するプレゼンテーションです。この動画で話されている内容のエッセンスを抽出し、それを箇条書きでまとめてください。話されている順番通りにまとめてください。"
    )
    summary = response.response
    return summary


@click.command()
@click.option("--video_url", type=str, required=False)
def main(video_url: str | None):
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="oshizo/sbert-jsnli-luke-japanese-base-lite")
    )

    llm = OpenAI(model="gpt-4", temperature=0, max_tokens=4096)
    prompt_helper = PromptHelper(
        context_window=4096,  # max_input_suze
        num_output=2048,  # num_output
        chunk_overlap_ratio=0.1,  # chunk_overlap_ratio
    )
    service_context = ServiceContext.from_defaults(
        llm=llm, prompt_helper=prompt_helper, embed_model=embed_model
    )
    set_global_service_context(service_context)

    loader = YoutubeTranscriptReader()

    video_df = pd.read_csv("data/remopoke_videos.csv")
    video_list = video_df["url"].tolist()
    summary_list: list[VideoSummary] = []

    if video_url is not None:
        summary = get_video_summary(loader, video_url)
        print(summary)
        return

    for video in tqdm(video_list):
        summary = get_video_summary(loader, video)
        summary_list.append(
            VideoSummary(
                title=video_df[video_df["url"] == video]["title"].values[0],
                url=video,
                summary=summary,
            )
        )

    video_df2 = pd.DataFrame(summary_list)
    video_df2.to_csv("output/remopoke_videos_summary2.csv", index=False)


if __name__ == "__main__":
    main()  # type: ignore

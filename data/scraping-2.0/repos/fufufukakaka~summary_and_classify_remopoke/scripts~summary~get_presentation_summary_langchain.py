import dataclasses
import os

import click
import openai
import pandas as pd
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from youtube_transcript_api._errors import TranscriptsDisabled

openai.api_key = os.environ["OPENAI_API_KEY"]


@dataclasses.dataclass
class VideoSummary:
    title: str
    url: str
    summary: str


def build_map_reduce_prompt_chain() -> MapReduceDocumentsChain:
    # Map
    map_template = """以下の文章はポケモンに関するプレゼンテーションの文字起こしです。:
    {docs}
    これらの文章を元にして、このプレゼンテーションの要点を自然な文章でまとめてください。
    わかりやすいまとめ:"""
    map_prompt = PromptTemplate.from_template(map_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """以下の文章はポケモンに関するプレゼンテーションの文字起こしから抽出された要点です。:
    {doc_summaries}
    これらの要点をまとめあげて、プレゼンテーション全体の要点を自然な文章でまとめてください。
    わかりやすいまとめ:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=8192,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    return map_reduce_chain


def get_video_summary(map_reduce_chain, video_url: str) -> str:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0.1
    )
    loader = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info=False,
        language=["ja"],
    )
    try:
        documents = loader.load()
    except TranscriptsDisabled:
        return ""
    split_docs = text_splitter.split_documents(documents)
    summary = map_reduce_chain.run(split_docs)
    return summary


@click.command()
@click.option("--video_url", type=str, required=False)
def main(video_url: str | None):
    map_reduce_chain = build_map_reduce_prompt_chain()

    video_df = pd.read_csv("data/remopoke_videos.csv")
    video_list = video_df["url"].tolist()
    summary_list: list[VideoSummary] = []

    if video_url is not None:
        summary = get_video_summary(map_reduce_chain, video_url)
        print(summary)
        return

    for video in tqdm(video_list):
        summary = get_video_summary(map_reduce_chain, video)
        summary_list.append(
            VideoSummary(
                title=video_df[video_df["url"] == video]["title"].values[0],
                url=video,
                summary=summary,
            )
        )

    video_df2 = pd.DataFrame(summary_list)
    video_df2.to_csv("output/remopoke_videos_summary_langchain_35turbo.csv", index=False)


if __name__ == "__main__":
    main()  # type: ignore

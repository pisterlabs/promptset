import os
from dataclasses import asdict

import pandas as pd
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm
from wandb.integration.langchain import WandbTracer

import wandb
from config import config


def get_data(artifact_name: str, total_episodes: int = None):
    podcast_artifact = wandb.use_artifact(artifact_name, type="dataset")
    podcast_artifact_dir = podcast_artifact.download(config.root_artifact_dir)
    filename = artifact_name.split(":")[0].split("/")[-1]
    df = pd.read_csv(os.path.join(podcast_artifact_dir, f"{filename}.csv"))
    if total_episodes is not None:
        df = df.iloc[:total_episodes]
    return df


def summarize_episode(episode_df: pd.DataFrame):
    # load docs into langchain format
    loader = DataFrameLoader(episode_df, page_content_column="transcript")
    data = loader.load()

    # split the documents
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    print(f"Number of documents for podcast {data[0].metadata['title']}: {len(docs)}")

    # initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # define map prompt
    map_prompt = """Write a concise summary of the following short transcript from a podcast.
    Don't add your opinions or interpretations.

    {text}

    CONCISE SUMMARY:"""

    # define combine prompt
    combine_prompt = """You have been provided with summaries of chunks of transcripts from a podcast.
    Your task is to merge these intermediate summaries to create a brief and comprehensive summary of the entire podcast.
    The summary should encompass all the crucial points of the podcast.
    Ensure that the summary is atleast 2 paragraph long and effectively captures the essence of the podcast.
    {text}

    SUMMARY:"""

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    # initialize the summarizer chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
    )

    summary = chain({"input_documents": docs})
    return summary


if __name__ == "__main__":
    # initialize wandb tracer
    WandbTracer.init(
        {
            "project": config.project_name,
            "job_type": "summarize",
            "config": asdict(config),
        }
    )

    # get scraped data
    df = get_data(artifact_name=config.yt_podcast_data_artifact)

    summaries = []
    with get_openai_callback() as cb:
        for episode in tqdm(df.iterrows(), total=len(df), desc="Summarizing episodes"):
            episode_data = episode[1].to_frame().T

            summary = summarize_episode(episode_data)
            summaries.append(summary["output_text"])

        print("*" * 25)
        print(cb)
        print("*" * 25)

        wandb.log(
            {
                "total_prompt_tokens": cb.prompt_tokens,
                "total_completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost,
            }
        )

    df["summary"] = summaries

    # save data
    path_to_save = os.path.join(config.root_data_dir, "summarized_podcasts.csv")
    df.to_csv(path_to_save, index=False)

    # log to wandb artifact
    artifact = wandb.Artifact("summarized_podcasts", type="dataset")
    artifact.add_file(path_to_save)
    wandb.log_artifact(artifact)

    # create wandb table
    table = wandb.Table(dataframe=df)
    wandb.log({"summarized_podcasts": table})

    WandbTracer.finish()

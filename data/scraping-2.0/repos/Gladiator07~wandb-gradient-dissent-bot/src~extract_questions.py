import os
import re
from dataclasses import asdict

import pandas as pd
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
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


def extract_questions(episode_df: pd.DataFrame):
    # load docs into langchain format
    loader = DataFrameLoader(episode_df, page_content_column="transcript")
    data = loader.load()

    # split the documents
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    print(f"Number of documents for podcast {data[0].metadata['title']}: {len(docs)}")

    # initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # define prompt
    prompt = """You are provided with a short transcript from a podcast episode.
    Your task is to extract the relevant and most important questions one might ask from the transcript and present them in a bullet-point list.
    Ensure that the total number of questions is no more than 3.

    TRANSCRIPT:

    {text}

    QUESTIONS:"""

    prompt_template = PromptTemplate(template=prompt, input_variables=["text"])

    pattern = r"\d+\.\s"
    que_by_llm = []
    for doc in docs:
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        out = llm_chain.run(doc)
        cleaned_ques = re.sub(pattern, "", out).split("\n")
        que_by_llm.extend(cleaned_ques)

    return que_by_llm


if __name__ == "__main__":
    # initialize wandb tracer
    WandbTracer.init(
        {
            "project": config.project_name,
            "job_type": "extract_questions",
            "config": asdict(config),
        }
    )

    # get data
    df = get_data(artifact_name=config.summarized_data_artifact)

    questions = []
    with get_openai_callback() as cb:
        for episode in tqdm(
            df.iterrows(), total=len(df), desc="Extracting questions from episodes"
        ):
            episode_data = episode[1].to_frame().T

            episode_questions = extract_questions(episode_data)
            questions.append(episode_questions)

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

    df["questions"] = questions

    # log to wandb artifact
    path_to_save = os.path.join(config.root_data_dir, "summarized_que_podcasts.csv")
    df.to_csv(path_to_save, index=False)
    artifact = wandb.Artifact("summarized_que_podcasts", type="dataset")
    artifact.add_file(path_to_save)
    wandb.log_artifact(artifact)

    # create wandb table
    df["questions"] = df["questions"].apply(lambda x: "\n".join(x))
    table = wandb.Table(dataframe=df)
    wandb.log({"summarized_que_podcasts": table})

    WandbTracer.finish()

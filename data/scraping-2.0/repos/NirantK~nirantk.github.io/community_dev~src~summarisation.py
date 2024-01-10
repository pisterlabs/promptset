import datetime
import json
import re
from functools import lru_cache
from pathlib import Path

import fire
import pandas as pd
import pytz
from formatting_utils import human_date
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from prompts import PROMPT_TEMPLATES

text_splitter = CharacterTextSplitter.from_tiktoken_encoder()


def make_docs(plain_text: str) -> list:
    texts = text_splitter.split_text(plain_text)
    docs = [Document(page_content=t) for t in texts]
    return docs


@lru_cache
def summarize_docs(
    docs: list,
    prompt_template: str,
    model,
    chain_type="stuff",
) -> str:
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # We should abstract chain logic when more chain type experiments are added
    if chain_type == "map_reduce":
        chain = load_summarize_chain(
            model, chain_type=chain_type, map_prompt=prompt, combine_prompt=prompt
        )
    else:
        chain = load_summarize_chain(model, chain_type=chain_type, prompt=prompt)
    chain_output = chain({"input_documents": docs}, return_only_outputs=True)
    return chain_output["output_text"]


def summarize(message: str, prompt_template: str, chain_type: str = "stuff") -> str:
    docs = make_docs(message)
    summary_text = summarize_docs(
        docs,
        prompt_template,
        chain_type="stuff",
        model=ChatOpenAI(temperature=0),
    )
    return summary_text


def extract_urls_context(text: str, window_size: int = 1) -> list:
    lines = text.split("\n")
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    urls_context = []

    for idx, line in enumerate(lines):
        for match in url_pattern.finditer(line):
            start, end = match.span()
            prev_line = lines[idx - window_size] if idx > 0 else ""
            next_line = lines[idx + window_size] if idx < len(lines) - 1 else ""
            context = f"{prev_line}\n{line}\n{next_line}".strip()
            # Dropping match.group() from append as we are not using it
            urls_context.append(context)
    return urls_context


# TODO: Below functions can be simplified and optimized if the purpose is more clearer
def get_page_header_date(date_object):
    # Combine the date object with a time object and set the desired timezone
    dt = datetime.datetime.combine(date_object, datetime.time())
    desired_timezone = pytz.timezone("Asia/Kolkata")
    localized_dt = desired_timezone.localize(dt)

    # Format the datetime object using strftime
    formatted_datetime = localized_dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    formatted_datetime = formatted_datetime[:-2] + ":" + formatted_datetime[-2:]

    return formatted_datetime


def make_page_header(row):
    date, summary_text = row["Date"], row["title_desc"]
    dt = get_page_header_date(date)
    fields = json.loads(summary_text)  # This is expensive!
    summary_title, summary_description = fields["title"], fields["description"]

    page_header = f"""+++
                    title =  "{summary_title}"
                    date = {dt}
                    tags = ["daily_summary"]
                    featured_image = ""
                    description = "{summary_description}"
                    toc = true
                    +++
                    """
    return page_header


def make_page(row):
    page = (
        row["page_headers"]
        + "\n"
        + row["Summary"]
        + "\n"
        + "\n## Links\nThe description and link can be mismatched because of extraction errors.\n\n"
        + row["EndNote"]
    )
    file_name = f"{human_date(row['Date'])}.md"
    return page, file_name


def generate_daily_df(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Date"] = df["Datetime"].dt.date
    daily_df = df.groupby("Date").agg({"Message": " \n ".join}).reset_index()
    daily_df["wc"] = daily_df["Message"].apply(lambda x: len(x.split()))
    return daily_df


def generate_daily_summary(csv_path: str) -> None:
    readpath = Path(csv_path).resolve()
    assert readpath.exists(), "CSV file does not exist"
    write_dir = Path("../../content/ai/").resolve()

    daily_df = generate_daily_df(readpath, True)
    # Generating the summary column
    daily_df["Summary"] = daily_df["Message"].apply(
        summarize, args=(PROMPT_TEMPLATES["summary_template"],)
    )

    # Generating the EndNote column
    daily_df["Endnote"] = (
        daily_df["Message"]
        .apply(extract_urls_context)
        .apply(
            lambda urls_context: "\n".join(
                [
                    summarize(message, PROMPT_TEMPLATES["link_context_template"])
                    for message in urls_context
                ]
            )
        )
    )

    # Generating Title and Description Columns that can be passed to header method
    # We are avoiding the for loop with this intermediate column
    daily_df["title_desc"] = daily_df["Summary"].apply(
        summarize,
        args=(
            PROMPT_TEMPLATES["title_description_template"],
            "map_reduce",
        ),
    )
    # Generating page headers
    page_headers = []
    for idx in range(len(daily_df)):
        page_headers.append(make_page_header(daily_df.iloc[idx]))

    # Dumping all the updates
    daily_df["page_headers"] = page_headers
    daily_df.to_json("daily_backup.json")  # This is always in the current directory

    # Using page headers to make pages
    for idx in range(len(daily_df)):
        page, file_name = make_page(daily_df.iloc[idx])
        file_path = write_dir / file_name
        with file_path.open("w") as f:
            f.write(page)


if __name__ == "__main__":
    fire.Fire(generate_daily_summary)

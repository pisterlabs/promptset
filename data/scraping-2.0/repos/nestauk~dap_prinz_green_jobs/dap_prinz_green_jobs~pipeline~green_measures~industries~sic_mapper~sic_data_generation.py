"""
This one-off script generates a SIC dataset and embeddings to map to that uses
LLMs to describe SIC codes as company descriptions.

python dap_prinz_green_jobs/pipeline/green_measures/industries/sic_mapper/sic_data_generation.py --production
"""
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

import numpy as np
import toolz
import ast
import os
from dotenv import load_dotenv
import pandas as pd
from typing import Tuple, Dict
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime

from dap_prinz_green_jobs import logger, BUCKET_NAME, PROJECT_DIR
from dap_prinz_green_jobs.getters.industry_getters import load_sic
from dap_prinz_green_jobs.getters.data_getters import save_to_s3
import yaml

from dap_prinz_green_jobs.utils.bert_vectorizer import BertVectorizer

load_dotenv()  # load the openAI key

if not os.getenv("OPENAI_API_KEY"):
    ValueError(
        "OPENAI_API_KEY not found in environment variables. Add key to .env file."
    )


# let's use a chat model with a larger context window to 'stuff' in
# the SIC code list chunks via prompting
chat_model = ChatOpenAI(model="gpt-3.5-turbo-16k")

# load config
config_path = os.path.join(PROJECT_DIR, "dap_prinz_green_jobs/config/base.yaml")
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def _create_prompt() -> ChatPromptTemplate:
    """
    Generates a prompt for the SIC name to company description task.
    """
    # system message prompt
    # this is very hacky because I don't actually want to pass an
    # input variable to the system message prompt, but I need to
    sys_prompt = PromptTemplate(
        input_variables=["job"],
        template="""You are an employer looking to recruit for {job}. Your goal is to describe SIC codes as company descriptions typically found in job adverts. You are given a list of SIC codes to describe as company descriptions.""",
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    sic_schema = [
        ResponseSchema(
            name="sic_name",
            description="The description of the SIC code that you are describing as a company description.",
        ),
        ResponseSchema(
            name="sic_company_description",
            description="Describe the SIC description as a company description found in a typical job advertisement. The company description should be 1-2 sentences long. Do not mention the SIC code in the company description.",
        ),
    ]

    # # parse the schema
    output_parser = StructuredOutputParser.from_response_schemas(sic_schema)
    format_instructions = output_parser.get_format_instructions()

    # # generate the prompt template
    hum_prompt = PromptTemplate(
        template="For every SIC code in a list of SIC codes, describe them as company descriptions typically found in job adverts.\n{format_instructions}\n The list of SIC codes to describe as company descriptions are in the following comma separated list: \n[{sic_codes_list}]\n",
        input_variables=["sic_codes_list"],
        partial_variables={"format_instructions": format_instructions},
    )
    sic_describer_prompt = HumanMessagePromptTemplate(prompt=hum_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, sic_describer_prompt]
    )

    return chat_prompt


def generate_sic_company_descriptions(
    sic_list_chunk: Tuple[str], chat_model: ChatOpenAI = chat_model
) -> str:
    """Generate company descriptions for SIC codes per
        SIC code list chunk.

    Args:
        sic_list_chunk (Tuple[str]): Tuple of SIC codes

    Returns:
        str: company descriptions for SIC codes in the chunk
    """
    chat_prompt = _create_prompt()
    chat_prompt_formatted = chat_prompt.format_messages(
        sic_codes_list=", ".join(sic_list_chunk), job="positions"
    )
    output = chat_model(chat_prompt_formatted)

    return output.content


def clean_llm_response(
    llm_response: str,
) -> Tuple[Dict[str, str]]:
    """Clean the LLM response to tuple of dictionaries
    where each dictionary has the sic_name and sic_company_description.

    Args:
        llm_response (str): LLM response

    Returns:
        Tuple[Dict[str, str]]: Tuple of dictionaries where each dictionary
            has the sic_name and sic_company_description.
    """

    llm_response_clean = (
        llm_response.replace("\n", "")
        .replace("\t", "")
        .replace("```", "")
        .replace("json", "")
        .replace("}{", "}, {")
        .replace("}}", "}")
    )
    llm_response_clean = ast.literal_eval(llm_response_clean)

    return llm_response_clean


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=50)
    # add a boolean production flag
    parser.add_argument("--production", action="store_true", default=False)

    args = parser.parse_args()

    chunk_size = args.chunk_size
    production = args.production

    # let's first load the SIC codes
    logger.info("loading SIC data to curate...")
    sic_data = load_sic()
    sic_names = dict(zip(sic_data["Most disaggregated level"], sic_data["Description"]))

    logger.info("curating dataset...")
    sic_df = pd.DataFrame(
        {"sic_code": list(sic_names.keys()), "sic_name": list(sic_names.values())}
    )
    sic_df_grouped = sic_df.groupby("sic_name").sic_code.apply(list).reset_index()

    # let's get the sic names that are unique to generate
    # company descriptions for
    logger.info(
        "Generating company descriptions for SIC codes...This will take an hour is production"
    )
    sic_names_unique = sic_df_grouped.sic_name.str.lower().tolist()
    # here if production is false, we will only generate company descriptions for sic codes of
    # chunk size n to test the pipeline
    sic_names_unique = (
        sic_names_unique[:chunk_size] if not production else sic_names_unique
    )
    print(len(sic_names_unique))
    sic_names_unique_chunks = toolz.partition_all(chunk_size, sic_names_unique)

    sic2compdescs = []
    for sic_name_chunk in tqdm(sic_names_unique_chunks):
        sic_comp_descs = generate_sic_company_descriptions(sic_name_chunk)
        sic_comp_descs_clean = clean_llm_response(sic_comp_descs)
        sic2compdescs.extend(sic_comp_descs_clean)

    # convert it this way so we can map to the dataset later
    sic2compdescs_dict = {
        i["sic_name"]: i["sic_company_description"] for i in sic2compdescs
    }

    logger.info("adding company descriptions to SIC dataset...")
    sic_df_grouped = sic_df_grouped.assign(
        sic_company_description=lambda x: x.sic_name.str.lower().map(
            sic2compdescs_dict
        ),
        id=lambda x: x.index,
    )[["id", "sic_code", "sic_name", "sic_company_description"]]

    sic_df_grouped["sic_code"] = sic_df_grouped.sic_code.apply(ast.literal_eval)
    sic_df_grouped.dropna(subset=["sic_company_description"], inplace=True)
    sic_df_grouped_dict = sic_df_grouped.to_dict(orient="records")

    logger.info("saving curated SIC dataset, SIC embedding and dictionary...")
    date_today = datetime.today().strftime("%Y-%m-%d").replace("-", "")
    sic_comp_desc_path = f"s3://{BUCKET_NAME}/outputs/data/green_industries/{date_today}_sic_company_descriptions_production_{production}_chunksize_{chunk_size}.csv"
    sic_df_grouped.to_csv(sic_comp_desc_path, index=False)
    save_to_s3(
        BUCKET_NAME,
        sic_df_grouped_dict,
        f"outputs/data/green_industries/{date_today}_sic_company_descriptions_dict_production_{production}_chunksize_{chunk_size}.json",
    )

    bert_model_name = f"sentence-transformers/{config['industries']['bert_model_name']}"
    bert_model = BertVectorizer(
        bert_model_name=bert_model_name,
        multi_process=config["industries"]["multi_process"],
    ).fit()

    sic_embeds = bert_model.transform(
        list(sic_df_grouped.sic_company_description.tolist())
    )
    # save the embeddings to s3
    sic_comp_desc_embeds_path = config["industries"]["sic_comp_desc_embeds_path"]
    data_outputs_path = config["job_adverts"]["data_folder_name"]

    save_to_s3(
        BUCKET_NAME,
        sic_embeds,
        os.path.join(data_outputs_path, "green_industries", sic_comp_desc_embeds_path),
    )

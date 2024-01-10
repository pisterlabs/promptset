import pathlib

from infra.supabase import supabase

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import create_structured_output_runnable
from rq import get_current_job


config = {
    "middle_group_mapping_model": "gpt-3.5-turbo",
    "narrow_group_mapping_model": "gpt-3.5-turbo",
    "nomenclature_mapping_model": "gpt-4-1106-preview",
    "nomenclature_description_model": "gpt-3.5-turbo",
}


description_template = "Что такое {input}"
description_prompt = PromptTemplate(
    template=description_template,
    input_variables=["input"]
)

group_template = """"Для объекта выбери строго одну подходящую категорию из списка.
Учти описание объекта.

Объект: {input}
Описание объекта: {description}

Категории:
{groups}
"""
group_prompt = PromptTemplate(
    template=group_template,
    input_variables=["input", "groups", "description"]
)

nomenclature_template = """Найди в списке 5 объектов с наиболее похожими названиями.

Объект: 
{input}

Список:
{groups}
"""
nomenclature_prompt = PromptTemplate(
    template=nomenclature_template,
    input_variables=["input", "groups"]
)

group_json_schema = {
    "type": "object",
    "properties": {
        "category": {"title": "category", "description": "category number and name", "type": "string"},
    },
    "required": ["category"],
}

nomenclature_json_schema = {
    "type": "object",
    "properties": {
        "nomenclature": {
            "type": "array",
            "description": "the most similar 5 objects",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["nomenclature"],
}

description_llm \
    = ChatOpenAI(temperature=0, model_name=config["nomenclature_description_model"], max_retries=3, request_timeout=30)
description_chain = LLMChain(llm=description_llm, prompt=description_prompt)


def create_runnable(model_name: str, schema: dict, prompt):
    llm = ChatOpenAI(temperature=0, model_name=model_name, max_retries=3, request_timeout=30)
    return create_structured_output_runnable(schema, llm, prompt)


def get_nomenclatures(group: str) -> str:
    response = supabase \
        .table("nomenclature") \
        .select("Номенклатура") \
        .eq("Группа", group) \
        .execute()
    return "\n".join([d["Номенклатура"] for d in response.data])


def get_groups(filepath: str, group_number: str = None) -> str:
    with open(filepath, "r") as f:
        lines = f.readlines()
        if group_number:
            lines = [line for line in lines if line.startswith(group_number)]
        return "".join(lines)


def map_with_groups(query: str, description: str, filepath: str, group_runnable, prev_response: str = None, index: int = None) -> str:
    groups = get_groups(filepath, prev_response[:index] if prev_response else None)

    if len(groups) == 0:
        return prev_response

    response = group_runnable.invoke({"input": query, "groups": groups, "description": description})
    return response["category"]


def map_with_nomenclature(query: str, final_group: str) -> list[str]:
    nomenclature_runnable \
        = create_runnable(config["nomenclature_mapping_model"], nomenclature_json_schema, nomenclature_prompt)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n"]
    )
    nomenclatures = get_nomenclatures(final_group)
    nomenclatures_chunks = text_splitter.split_text(nomenclatures)
    inputs = [{"input": query, "groups": c} for c in nomenclatures_chunks]
    responses = nomenclature_runnable.batch(inputs, {"max_concurrency": 6})
    short_list = []
    for r in responses:
        short_list.extend(r["nomenclature"])
    groups = "\n".join(str(short_list))
    response = nomenclature_runnable.invoke({"input": query, "groups": groups})
    return response["nomenclature"]


def get_data_folder_path():
    return f"{pathlib.Path(__file__).parent.resolve()}"


def save_to_database(values: dict):
    return supabase.table("nomenclature_mapping").insert(values).execute()


def do_mapping(query: str, use_jobs: bool = True) -> list[str]:
    with get_openai_callback() as cb:
        description = description_chain.run(query)
        if use_jobs:
            job = get_current_job()

        middle_group = map_with_groups(
            query,
            description,
            f"{get_data_folder_path()}/../data/parent-parent.txt",
            create_runnable(config["middle_group_mapping_model"], group_json_schema, group_prompt)
        )
        if use_jobs:
            job.meta["middle_group"] = middle_group
            job.save_meta()

        narrow_group = map_with_groups(
            query,
            description,
            f"{get_data_folder_path()}/../data/parent.txt",
            create_runnable(config["narrow_group_mapping_model"], group_json_schema, group_prompt),
            middle_group,
            6
        )
        if use_jobs:
            job.meta["narrow_group"] = narrow_group
            job.save_meta()

        response = map_with_nomenclature(query, narrow_group)
        if use_jobs:
            database_response = save_to_database({
                "input": query,
                "output": ", ".join(response),
                "wide_group": "",
                "middle_group": middle_group,
                "narrow_group": narrow_group,
                "source": job.get_meta().get("source", None),
                "status": job.get_status()
            })

        if use_jobs:
            job.meta["mapping_id"] = database_response.data[0]["id"]
            job.save()

    # print(cb)

    return response


if __name__ == "__main__":
    print("FINAL_RESPONSE", do_mapping("Щит навесной распределительный ЩРН-12 IP31 250х300х120мм", use_jobs=False))

from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import StructuredOutputParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd


def layer_1(
    user_input,
    openai_api_key,
    response_schemas,
    system_message,
    human_message,
):
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=human_message,
            input_variables=["symptoms"],
            partial_variables={"format_instructions": format_instructions},
        )
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [human_message_prompt, system_message_prompt]
    )
    layer_1_chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = layer_1_chain.run({"symptoms": user_input})
    result_layer_1 = output_parser.parse(result)
    return result_layer_1


def get_similarity_search_from_pinecone(
    index_name, layer_1_suggestions, openai_api_key
):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace="divide_by_diseases"
    )
    docs = docsearch.similarity_search(layer_1_suggestions, k=1)
    return docs


def non_refine_target(
    disease_input,
    openai_api_key,
    response_schemas,
    system_message,
    human_message,
):
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=human_message,
            input_variables=["Disease"],
            partial_variables={"format_instructions": format_instructions},
        )
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [human_message_prompt, system_message_prompt]
    )
    non_refine_chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = non_refine_chain.run({"Disease": disease_input})
    result_non_refine = output_parser.parse(result)
    return result_non_refine


def layer_2(
    docs,
    query,
    layer_1_suggestions,
    openai_api_key,
    response_schemas,
    refine_prompt_template,
    initial_qa_template,
):
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_prompt_template,
    )
    initial_qa_prompt = PromptTemplate(
        input_variables=["context_str", "existing_answer"],
        template=initial_qa_template,
        partial_variables={"format_instructions_layer_2": format_instructions},
    )

    layer_2_chain = load_qa_chain(
        llm=chat,
        chain_type="refine",
        return_refine_steps=True,
        question_prompt=initial_qa_prompt,
        refine_prompt=refine_prompt,
    )
    result_layer_2_1 = layer_2_chain(
        inputs={
            "input_documents": [docs[0]],
            "question": query,
            "existing_answer": layer_1_suggestions,
        },
        return_only_outputs=True,
    )
    result_layer_2 = output_parser.parse(result_layer_2_1["output_text"])
    return result_layer_2


def create_df_food_nutrient(db):
    food_rows = []
    nutrient_rows = []
    docs_food = db.collection("CatFood").stream()
    docs_nutrient = db.collection("CatNutrient").stream()
    for i in docs_food:
        food_rows.append(i.to_dict())
    for j in docs_nutrient:
        nutrient_rows.append(j.to_dict())
    df_catfood = pd.DataFrame(food_rows)
    df_nutrients = pd.DataFrame(nutrient_rows)
    return df_catfood, df_nutrients


def get_CatFood(df, nutrient_list, top):
    nutrient_dict = {
        "Protein": {"High": 0.4, "Moderate": 0.46, "Low": 0.45, "None": 0},
        "Fat": {"High": 0.23, "Moderate": 0.23, "Low": 0.16, "None": 0},
        "Carbs": {"High": 0.2, "Moderate": 0.2, "Low": 0.1, "None": 0},
        "Fiber": {"High": 0.02, "Moderate": 0.02, "Low": 0.018, "None": 0},
        "Moisture": {"High": 0.1, "Moderate": 0.1, "Low": 0.09, "None": 0},
        "level": {"High": ">=", "Moderate": "<=", "Low": "<=", "None": ">="},
    }
    filter_expr = ""
    level = ""
    value = 0
    for i in nutrient_list:
        level = nutrient_dict["level"][i[1]]
        value = nutrient_dict[i[0]][i[1]]
        filter_expr += f"(df['{i[0]}']{level}{value}) &"
    filter_expr = filter_expr[:-3] + ")"
    filtered_df = (
        df[eval(filter_expr)].sort_values("Protein", ascending=False).head(top)
    )
    return filtered_df


def review_table(db):
    review_rows = []
    docs_review = db.collection("Reviews").stream()
    for review in docs_review:
        review_rows.append(review.to_dict())
    df_review = pd.DataFrame(review_rows).sort_values("datetime")
    return df_review

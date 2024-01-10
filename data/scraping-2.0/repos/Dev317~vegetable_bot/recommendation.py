from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from typing import List
import json


class RecommendationItem(BaseModel):
    product_name: str = Field(description="The name of the recommended product")
    product_explanation: str = Field(description="The explanation for the recommendation")
    product_image_link: str = Field(description="The link to the product image")
    product_link: str = Field(description="The link to the product")
    product_price: str = Field(description="The price of the product")

class Recommendation(BaseModel):
    recommendation: List[RecommendationItem]


def get_parser():
    return PydanticOutputParser(pydantic_object=Recommendation)

def parse_output(response):
    recommended_products = json.loads(response['result'])['recommendation']

    image_list = []
    product_list = []
    for idx, product in enumerate(recommended_products):
        product_description = f"""{idx+1}. Sản phẩm: {product['product_name']}\n
Giá thành: {product['product_price']}\n
Mô tả: {product['product_explanation']}\n
Link sản phẩm: {product['product_link']}\n
"""

        product_list.append(product_description)
        image_list.append(product['product_image_link'])
    return product_list, image_list

def get_chain():
    parser = get_parser()
    df = pd.read_csv("vinh_ha_dataset.csv")

    product = """
    Loại Sản Phẩm: {product_category}
    Sản Phẩm: {product_name}
    Giá Thành: {product_price}
    Link Sản Phẩm: {product_link}
    Mô Tả: {product_description}
    Link Ảnh: {product_image_link}
    """

    product_list = []
    for index, row in df.iterrows():
        product_list.append(product.format(
            product_category=row["Loại Sản Phẩm"],
            product_name=row["Tên Sản Phẩm"],
            product_price=row["Giá Thành"],
            product_link=row["Link Sản Phẩm"],
            product_description=row["Mô Tả Đầy Đủ"],
            product_image_link=row["Link Ảnh"]
        ))

    template = """"
    Dựa vào tên, giá thành, link và mô tả của những sản phẩm dưới đây:
    {context}

    Hãy giới thiệu những sản phẩm, đi kèm với Giá Thành, Link sản phẩm và Link ảnh dựa theo những yêu cầu sau đây:
    {format_instructions}\n{question}\n"""


    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    model_name = "gpt-3.5-turbo-16k"

    llm = ChatOpenAI(model=model_name)

    embeddings = OpenAIEmbeddings(model=model_name)
    retriever = Chroma.from_texts(product_list, embedding=embeddings).as_retriever()


    chain_type_kwargs = {"prompt": prompt}

    return RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)

def get_response(input):

    # Ex: "Có những loại sản phẩm nào dưới mức giá 60,000 VND mà có thể chế biến nhanh và phù hợp cho những người muốn giảm cân"
    qa = get_chain()
    result = qa({"query": input})
    return parse_output(result)

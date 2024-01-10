from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import camelot.io as camelot
from dotenv import load_dotenv
import os
import pathlib
import io
from io import StringIO

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

##### Schema #####

estate_schema = Object(
    id="estate_schema",
    description="""
        入札イベントの日時と場所。
    """,
    attributes=[
        Text(
            id="price",
            description="物件の価格"
        ),
        Text(
            id="location",
            description="物件の所在地"
        ),
        Text(
            id="structure",
            description="物件の構造"
        ),
        Number(
            id="floors",
            description="物件の階建"
        ),
        Number(
            id="floor_part",
            description="物件の階部分"
        ),
        Number(
            id="area",
            description="物件の面積"
        )
    ],
    examples=[
        (
            " 価格 １，６７０万円 所在地         東京都台東区上野7丁目8-15        構造 鉄筋コンクリート造 9階建 6階部分  専有面積 ２５．５８㎡",
            [
                {"price": "１，６７０万円", "location": "東京都荒川区荒川一丁目28-3", "structure": "鉄筋コンクリート造", "floors": 9, "area": "２５．５８㎡",
                 "floor_part": 6},
            ]
        ),
        (
            "価格  ２，５２０万円  所在地       神奈川県川崎市川崎区藤崎１－３－４  鉄骨鉄筋コンクリート造　地上15階建　8階部分 専有面積 19.67㎡",
            [
                {"price": "２，５２０万円", "location": "神奈川県川崎市川崎区藤崎１－３－４", "structure": "鉄骨鉄筋コンクリート造", "floors": 15, "area": "19.67㎡",
                 "floor_part": 8},
            ]
        ),
    ]
)

DICT_SCHEMA = {"price": "価格", "location": "所在地", "structure": "構造", "floors": "階建", "floor_part": "階部分", "area": "面積"}


def change_key(dict_output, dict_key):
    temp_dict = dict()
    keys = list(dict_output.keys())
    for item in keys:
        if item in dict_key:
            temp_dict[dict_key[item]] = dict_output[item]
        else:
            temp_dict[item] = dict_output[item]
    return temp_dict


def main():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",  # "gpt-3.5-turbo"
        temperature=0,
        max_tokens=4096,
        openai_api_key=openai_api_key
    )
    st.set_page_config(page_title="Extract structured real estate data from pdf file")
    st.header("Extract structured real estate data from pdf file 💬")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        pathlib.Path('temp.pdf').write_bytes(bytes_data)

        tables = camelot.read_pdf('temp.pdf')
        # text = tables[0].df.to_string()
        output = tables[0].df.to_string(header=False, index=False)
        text = " ".join(output.split())
        st.write("Input")
        st.write(text)

        chain = create_extraction_chain(llm, estate_schema, encoder_or_encoder_class='json')
        with get_openai_callback() as cb:
            output = chain.predict_and_parse(text=text)["data"]
        response_output = change_key(output['estate_schema'][0], DICT_SCHEMA)
        st.write("Output")
        st.write(response_output)
        st.write("Cost")
        st.write(cb)


if __name__ == '__main__':
    main()

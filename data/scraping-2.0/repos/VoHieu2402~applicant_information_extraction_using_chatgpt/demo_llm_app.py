import streamlit as st
import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import xlsxwriter
import openai
from langchain.chat_models import ChatOpenAI
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from pypdf import PdfReader

st.write("""            
# From unstructured data to structured data
## Demo by Huge Vo, Dean Le, Panukadu and Ant Vo

This app helps map information extracted from CV files to specific data fields
""")

lst_pd_cv = []
uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True, type=["pdf"])

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

OPENAI_API_KEY = st.text_input(
        "Enter your OpenAI key 👇",
    )
if OPENAI_API_KEY:
    st.write("You entered: ", OPENAI_API_KEY)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    schema = Object(
        id="CV",
        description="Information about a candidate presented by a resume",
        attributes=[
            Text(
                id="full_name",
                description="The full name of the candidate",
                examples=[("John Smith wants to be a doctor", "John Smith")],
            ),
            Text(
                id="phone_number",
                description="The phone number of the candidate",
                examples=[("Mobile: (+84) 822 540 983", "(+84) 822 540 983"), ("Phone: 0901458047", "0901458047")],
            ),
            Text(
                id="email",
                description="The email of the candidate",
                examples=[("Email: vominhhieu@gmail.com", "vominhhieu@gmail.com")],
            ),
            Text(
                id="university",
                description="The university of the candidate",
                examples=[("Education: Foreign Trade Univerity", "Foreign Trade University")]
            ),
            Text(
                id="major",
                description="The major of the candidate in university",
                examples=[(
                    '''
                    Education
                    Foreign Trade University
                    International Finance
                    ''',
                    "International Finance"
                )]
            ),
            Text(
                id="experiences",
                description="The working experiences the candidate has gained",
                examples=[(
                    '''
                    Experiences
                    2021-2022
                    Soho Academy
                    Ielts tutor
                    teach ielts to students online
                    advise and mark student's test
                    ''',
                    "Soho Academy - Ielts tutor - teach ielts to students online, advise and mark student's test"
                )]
            ),
            Number(
                id="gpa",
                description="The gpa score of the candidate in Foreign Trade University",
                examples=[(
                    '''
                    Education
                    Foreign Trade University
                    GPA: 3.5
                    High school for the gifted
                    GPA: 8.0
                    ''',
                    3.5
                )]
            )
        ],
        examples=[
            (
                '''
                Vo Doan Hoang Anh
                429/38/2 Chien Luoc,Binh Tri Dong A ward, Binh Tan District, Ho Chi Minh city, Vietnam
                Mobile: (+84) 822 540 983
                Email: vodoanhoanganh.sesc@gmail.com
                EDUCATION
                FOREIGN TRADE UNIVERSITY
                Banking
                100{%} English-taught study program
                Current GPA: 3.18 out of 4.0
                Member of Communication Department of Securities Studying Club since December 2020
                Member of Communication Department of FTU’s Day event from September 2020 - November 2020
                Experiences
                2021-2022
                KPMG
                Audit intern
                collect data and assess credit risks
                ''',
                [
                    {"full_name": "Vo Doan Hoang Anh", "phone_number": "(+84) 822 540 983", "email": "vodoanhoanganh.sesc@gmail.com", "university": "FOREIGN TRADE UNIVERSITY", "major": "Banking", "gpa": 3.18, "experiences":"KPMG - Audit Intern - collect data and assess credit risks"},
                ],
            ),
            (
                '''
                Vo Doan Hoang Anh
                429/38/2 Chien Luoc,Binh Tri Dong A ward, Binh Tan District, Ho Chi Minh city, Vietnam
                Email: vodoanhoanganh.sesc@gmail.com
                EDUCATION
                FOREIGN TRADE UNIVERSITY
                100{%} English-taught study program
                Current GPA: 3.18 out of 4.0
                Member of Communication Department of Securities Studying Club since December 2020
                Member of Communication Department of FTU’s Day event from September 2020 - November 2020
                ''',
                [
                    {"full_name": "Vo Doan Hoang Anh", "phone_number": None, "email": "vodoanhoanganh.sesc@gmail.com", "university": "FOREIGN TRADE UNIVERSITY", "major": None, "gpa": 3.18, "experiences":None},
                ],
            )
        ],
        many=True,
    )
    chain = create_extraction_chain(llm, schema)


    if uploaded_files:
        for uploaded_file in uploaded_files:
            reader = PdfReader(uploaded_file)
            number_of_pages = len(reader.pages)
            input = ""
            for i in range(number_of_pages):
                page = reader.pages[i]
                text = page.extract_text()
                input += text
            response = chain.run(input)
            df = pd.DataFrame(response["data"]["CV"])
            lst_pd_cv.append(df)
            
        df_result = pd.concat(lst_pd_cv)
        st.download_button(
            label="Download data as CSV",
            data=to_excel(df_result),
            file_name='cv_info.xlsx',
        )
        st.table(df_result)

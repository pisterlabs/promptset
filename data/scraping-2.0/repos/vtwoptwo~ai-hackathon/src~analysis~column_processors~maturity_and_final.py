import json
from typing import Optional
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from .helpers.utils import (
    instruction_response,
    output_format_checker,
    generate_full_prompt,
    retry_on_rate_limit_error,
    instruction_response_with_max_tokens,
)
import re
import ast
from datetime import datetime


@output_format_checker(max_attempts=5, desired_format=list)
@retry_on_rate_limit_error(wait_time=10)
def get_maturity_and_final_date(document_name: str) -> Optional[list]:
    # Launch date

    with open(document_name) as f:
        data = json.load(f)

    documents = data["full_text"]
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0.3, separator=" "
    )
    texts = text_splitter.split_text(documents)
    embeddings = OpenAIEmbeddings(
    )
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    final_date = retriever.get_relevant_documents(
        "Maturity Date",
    )

    prompt2 = f"""
    Take a deep breath and think step by step
    Make sure you get this right, my life depends on it!
    Extract the maturity date and the final valuation date of the term sheet. 
    You have to know that the final valuation date is always before the maturity date in time. So if the 
    maturity date is '01/01/2022', the final valuation date is always before that date. Also, the dates cant be the same, so 
    if the maturity date is '01/01/2022', the final valuation date cant be '01/01/2022'.

    Given the following context,  extract the maturity date and the final valuation date of the term sheet.
    Read carefully, my life depends on it.
    Dont confuse both terms
    Your response format example: ['dd/mm/yyyy', 'dd/mm/yyyy'], where the first date is the final valuation date and the second date is the maturity date. 
    I just want the date, not the rest of the text. Dont create functions and staff, and follow the example format.
    Here is the context: {final_date}
    #Example: ['01/01/2021', '01/01/2022']
    Do not use written dates!
    ##Output
    Final valuation date and Maturity date:
    """

    final_date = instruction_response_with_max_tokens(prompt2, temp=0.3).strip()

    pattern = r"\d{2}/\d{2}/\d{4}"
    date_list = re.findall(pattern, final_date)
    if len(date_list) == 2:
        str_final_date_1 = datetime.strptime(date_list[0], "%d/%m/%Y")
        str_final_date_2 = datetime.strptime(date_list[1], "%d/%m/%Y")
        if str_final_date_1 > str_final_date_2 or str_final_date_1 == str_final_date_2:
            return None
        else:
            return date_list

    return final_date

#import os
#os.environ["OPENAI_API_KEY"] = "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" # thay key gpt vao day
from openai import OpenAI
import fitz
import pandas
from dotenv import load_dotenv,find_dotenv
import os
load_dotenv(find_dotenv(),override=True)
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
client = OpenAI(api_key = OPEN_AI_API_KEY)
from Retrive_weaviate import List_of_meta_data

def extract_tables(metadatas: list[dict]):
    tables = []
    i = 0
    for metadata in metadatas:
        
        source, page, _, _, _, _ = metadata.values()
        source_type = source.split('.')[-1]
        if source_type == 'pdf':
            page = int(page)
            doc = fitz.open(source)
            table = doc.load_page(page).get_text()
            tables.append(f'context {i}: {table}\n')
            i = i+1
        if source_type == 'xlsx':
            table = f'context {i}:'
            data = pandas.ExcelFile(source)
            for sheetname in data.sheet_names:
                df = pandas.read_excel(data, sheet_name=sheetname).fillna('').values.astype(str)
                df = [','.join(val) for val in df]
                df = '\n'.join(df)
                table += '\n' + df
            tables.append(table)
            i = i+1
    return ''.join(tables)

def answering(query, tables):
    question = f"""
    You are an assistant for question-answering tasks.
    Use one of the three following pieces of retrieved contexts to answer the question. If you don't know the answer, just say that you don't know. Show steps to solve it.
    Answer in Vietnamese
    Question: {query} 
    Context: {tables} 
    Answer:
    
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature= 0,
        
        messages=[
            {"role": "system", "content": question},
        ]
    )
    return response.choices[0].message.content

def answering_with_metadata(query, metadatas):
    tables = extract_tables(metadatas)
    return answering(query, tables)

if __name__ == "__main__":
    query = "Làm thế nào để nạp ti nơ vào bình trên máy VDCM0001.2.3.a "
    metadatas = List_of_meta_data(query=query)
    print(metadatas)
    list_of_metadata = []
    for meta in metadatas:
        source = {
            "source": meta["source"],
            "page": meta["page"]
        }
        list_of_metadata.append(source)
    print(list_of_metadata)
    #addon = extract_tables(metadatas)
    #answer = answering(query, extract_tables(metadatas))
    #print(answer)

#HÀM ĐÓNG GÓI
def Response(query):
    metadatas = List_of_meta_data(query=query)
    list_of_metadata = []
    for meta in metadatas:
        source = {
            "source": meta["source"],
            "page": meta["page"]
        }
        list_of_metadata.append(source)
    answer = answering(query, extract_tables(metadatas))
    return answer,list_of_metadata
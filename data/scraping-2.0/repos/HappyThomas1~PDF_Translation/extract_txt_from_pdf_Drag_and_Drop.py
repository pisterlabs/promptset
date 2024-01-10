import logging
import os.path
import zipfile
import json
import ijson
import openai
import time
import re
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, IntVar
from tkinterdnd2 import TkinterDnD
import PyPDF2
import tempfile


from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation

# Configuration
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# Constants
CHATGPT_MODEL = "gpt-3.5-turbo"
CHATGPT_URL =  "https://api.openai.com/v1/chat/completions"
MAX_TOKEN = 8192
TEMPERATURE = 0.7
credential_path = "PDF_credential.json"
json_file ="structuredData.json"



def extract_pdf_text(pdf_file):
    try:
        # Initial setup, create credentials instance.
       
        credentials = \
            Credentials.service_principal_credentials_builder().\
                with_client_id(os.getenv('PDF_SERVICES_CLIENT_ID')).with_client_secret(os.getenv('PDF_SERVICES_CLIENT_SECRET')).build() # type: ignore

        # Create an ExecutionContext using credentials and create a new operation instance.
        execution_context = ExecutionContext.create(credentials)
        extract_pdf_operation = ExtractPDFOperation.create_new()

        # Set operation input from a source file.
        source = FileRef.create_from_local_file(pdf_file)
        extract_pdf_operation.set_input(source)

        # Build ExtractPDF options and set them into the operation
        extract_pdf_options = ExtractPDFOptions.builder().with_element_to_extract(ExtractElementType.TEXT).build()
        extract_pdf_operation.set_options(extract_pdf_options)

        # Execute the operation.
        result = extract_pdf_operation.execute(execution_context)
        
        return result
    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        logging.exception("Exception encountered while executing operation")
        raise e  # この行を追加します



def translate_with_chatgpt(text):

    prompt = f"""医学論文の一節です。日本語に翻訳してください。翻訳文のみ表示してください。 \n\n文章:\n  {text}"""
    data = create_chat_completion_with_retry(prompt, retries=3, delay=5)        
    translated_text = data["choices"][0]["message"]["content"].strip() # type: ignore
    print(translated_text)
    return translated_text

def create_chat_completion_with_retry(prompt, retries=3, delay=5):
    for _ in range(retries):
        try:
            data = openai.ChatCompletion.create(
                model= CHATGPT_MODEL,
                messages= [{"role": "user", "content": prompt}]
            )
            return data
        except Exception as e:  
            print(f"Request failed with {e}, retrying...")
            time.sleep(delay)  # Wait for 'delay' seconds before next retry

    # If control reaches here, all retries have failed
    return  {
        "choises": [{"message": {"content" : "Error was not resolved!"}}]
    }




def write_to_file(file_path, text):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text)

def read_file_in_chunks(file_path, max_chars=2000):
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        accumulated_chars = 0
        for line in f:
            line_chars = len(line)
            if accumulated_chars + line_chars > max_chars:
                # もし行を追加すると最大文字数を超えるならば、その行を分割する
                last_period_index = line.rfind('. ')
                
                if last_period_index != -1:
                    # ピリオドが存在する場合はそこで行を二つに分けます
                    chunk.append(line[:last_period_index + 1])
                    yield chunk
                    # 新しいチャンクを開始し、行の残りを新しいチャンクに追加する
                    chunk = [line[last_period_index + 1:]]
                    accumulated_chars = len(chunk[0])
                else:
                    # ピリオドが存在しない場合は、新しいチャンクを開始する
                    yield chunk
                    chunk = [line]
                    accumulated_chars = line_chars
            else:
                # 行を追加しても最大文字数を超えない場合、その行を現在のチャンクに追加する
                chunk.append(line)
                accumulated_chars += line_chars

        if chunk: 
            # 最後のチャンクを返す
            yield chunk

def copy_tmp_file(translated_text_path):
    # 現在の日付と時刻を取得
    now = datetime.now()
    # ファイル名に現在の日付と時刻を組み込む（フォーマットは任意）
    new_file_name = f"translated_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    # ファイルをコピー
    shutil.copy2(translated_text_path, os.path.join(os.path.dirname(translated_text_path),new_file_name))

def translate_pdf(input_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    zip_file = os.path.join(output_path ,"text.zip")
    try:
        if not os.path.exists(zip_file):
            pdf_text = extract_pdf_text(input_path)
            pdf_text.save_as(zip_file)
    except (ServiceApiException, ServiceUsageException, SdkException):
        logging.exception("Exception encountered while executing operation")

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Unzip all files and save
        zip_ref.extractall(output_path)

    json_path = os.path.join(output_path,json_file)
    raw_text_path = os.path.join(output_path, "Rawtext.txt")
    translated_text_path = os.path.join(output_path, "translated_text.txt")

    with open(json_path, 'r', encoding= 'utf-8') as f:
        objects = ijson.items(f, 'elements.item')
        with open(raw_text_path, 'w', encoding='utf-8') as text_file:
            text_file.write('\n'.join(re.sub('\(<https://[^>]*>\)', '', item["Text"]) for item in objects if 'Text' in item))


    with open(translated_text_path,'w') as f:
        f.close()

    with open(translated_text_path,'a', encoding ='utf-8') as f:
        for chunk in read_file_in_chunks(raw_text_path):
            translated_text = translate_with_chatgpt(chunk)
            f.write(translated_text)
    
    copy_tmp_file(translated_text_path)


def extract_pages(input_path, page_numbers):
    with open(input_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        writer = PyPDF2.PdfWriter()

        for page_number in page_numbers:
            page = reader.pages[page_number]
            writer.add_page(page)

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        with open(temp_pdf.name, 'wb') as output_file:
            writer.write(output_file)

    return temp_pdf.name

def translate_selected_pages(input_path, output_path, page_numbers):
    temp_pdf_path = extract_pages(input_path, page_numbers)
    translate_pdf(temp_pdf_path, output_path)
    os.remove(temp_pdf_path)

def on_drop(event):
    pdf_path = event.data.strip()

    output_path = pdf_path[:-4] + '_output'
    start_page = int(start_page_entry.get()) - 1
    end_page = int(end_page_entry.get())
    page_numbers = list(range(start_page, end_page))

    translate_selected_pages(pdf_path, output_path, page_numbers)

    label.config(text='Translation saved as ' + output_path)
    root.destroy()

def main():
    global label, start_page_entry, end_page_entry, root

    root = TkinterDnD.Tk()
    root.geometry('400x300')
    root.title('PDF Translator')

    label = tk.Label(root, text='Drag and drop a PDF file to translate', pady=20)
    label.pack()

    root.drop_target_register('*')
    root.dnd_bind('<<Drop>>', on_drop)

    start_page_label = tk.Label(root, text='Start page:')
    start_page_label.pack()

    start_page_entry = tk.Entry(root)
    start_page_entry.pack()

    end_page_label = tk.Label(root, text='End page:')
    end_page_label.pack()

    end_page_entry = tk.Entry(root)
    end_page_entry.pack()

    root.mainloop()





if __name__ == "__main__":
    main()

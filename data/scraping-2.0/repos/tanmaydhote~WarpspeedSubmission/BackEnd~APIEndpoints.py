from fastapi import FastAPI, File, UploadFile
import requests
import io
from langchain.tools import YouTubeSearchTool
from pypdf import PdfReader

from google.cloud import vision
import io
client = vision.ImageAnnotatorClient()

app = FastAPI()


@app.get("/hello")
async def hello(name:str):
   modified_name = name + "777"
   return modified_name


@app.get("/getYoutubeLinks")
async def search(query:str):
    tool = YouTubeSearchTool()
    results = tool.run(query,5)
    return results

@app.post("/processPDF")
async def read_pdf(file: UploadFile):
    contents = await file.read()
    text_dict = {}
    with io.BytesIO(contents) as data_stream:
        reader = PdfReader(data_stream)
        number_of_pages = len(reader.pages)
        file_text = ""
        text = ""
        # for page in range(number_of_pages):
        #     page_obj = reader.pages[page]
        #     text += page_obj. extract_text()
        print(number_of_pages)
        if(number_of_pages <= 30):
            pages = reader.pages
            for i in range(len(reader.pages)):
                page = reader.pages[i]
                s = page.extract_text()
                text = text + s
        return text

@app.post("/processImage")
async def get_OCR(file: UploadFile):
    content = await file.read()
    image = vision.Image(content=content)
    full_text = ""
    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    word_text = word_text.replace(" ", "|")
                    full_text = full_text + word_text + " "

    return full_text
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas 
from dotenv import find_dotenv, load_dotenv
import openai
import os


def convert_text_to_pdf(text):

    output_pdf_file = 'output.pdf'
    c = canvas.Canvas(output_pdf_file)
    # c.setFont('Helvetica', 12)
    y = 700
    line_height = 14

    for line in text.split('\n'):
        c.drawString(50,y,line)
        y -= line_height

    c.save()


def load_key():
    _ = load_dotenv(find_dotenv())

    openai_key = os.getenv("openai_key")
    os.environ["OPENAI_API_KEY"] = openai_key

    openai.api_key = openai_key #openai pure API not langchain one won't work with environ variables

    huggingface_key = os.getenv("huggingface_key")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key
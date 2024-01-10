from io import BytesIO
import PyPDF2
import urllib
import openai
from textblob import TextBlob
import csv



def convert_one_pdf(pdf_file):

    ## from URL 

    # url="https://www.telstra.com.au/content/dam/tcom/help/critical-information-summaries/personal/mobile/mobile-plans/Telstra-Upfront-Mobile-Plans-.pdf"

    # response = urllib.request.urlopen(url)
    # pdf_file = BytesIO(response.read())

    
    pdfReader = PyPDF2.PdfReader(pdf_file)
    number_of_pages = len(pdfReader.pages)

    text = ""
    for pageNum in range(number_of_pages):
        text += pdfReader.pages[pageNum].extract_text()

    # print(text)

    # ## write to a file
    # open the file in the write mode
    file = open('../../data/bills_minmin.txt','a')
    file.write("\"")
    file.write(text.replace("\n"," "))
    file.write("\"")
    file.write("|")



    # ## call opanAI to get the embedded value
    openai.api_key = "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"
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text)
    embedding=response['data'][0]["embedding"]

    print(embedding)

    # # ## write to a file
    # file = open('../../data/postpaid_mobile_plan_embedding.txt','a')
    file.write("\"")
    file.write(str(embedding))
    file.write("\"")

    file.close()

    # # # open the file in the write mode
    # with open('../../data/postpaid_mobile_plan_embedding.csv', 'w') as f:
    #     # create the csv writer
    #     writer = csv.writer(f)

    #     # write a row to the csv file
    #     writer.writerow(str(embedding))
    #     f.close()


def count_tokens(vCountTokenStr):
    # Tokenize the input string
    blob = TextBlob(vCountTokenStr)
    tokens = blob.words

    # Count the number of tokens
    num_tokens = len(tokens)
    return num_tokens

def fit_within_token_limit(text, token_limit):
    remaining_tokens = token_limit
    shortened_text = text

    while count_tokens(shortened_text) >token_limit:
        # Reduce the length of the text by 10% and try again
        shortened_length = int(len(shortened_text) * 0.9)
        shortened_text = shortened_text[:shortened_length]

    return shortened_text



file1 = open(r'../../data/Mobile Bill_MinminDu_202307.pdf', mode='rb')
convert_one_pdf(file1)


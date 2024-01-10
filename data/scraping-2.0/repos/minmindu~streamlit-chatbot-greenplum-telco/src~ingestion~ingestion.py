from minio import Minio
from io import BytesIO
import PyPDF2
import urllib
import openai
from textblob import TextBlob
import csv


minio_host_port="35.189.1.42:9000"
minio_access_key="minioadmin"
minio_secret_key="minioadmin"
minio_bucket_name="telco-bills"
file_prefix="invoice"
# file_prefix="invoice21000202107"

openai_api_key = "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"

save_file_path="../../data/invoice_embedding.csv"

'''
This function will connect to MinIO server, 
Return: minio connection
'''
def connect_minio():
    
    ## Note: secure=False is required for HTTP, not HTTPS
    client = Minio(minio_host_port, 
                secure=False, 
                access_key=minio_access_key,
                secret_key=minio_secret_key)
    
    return client



'''
This function will convert a rb pdf content to text
Return: the pdf content::text
'''
def get_pdf_content(pdf_file):

    ## from URL 

    # url="https://www.telstra.com.au/content/dam/tcom/help/critical-information-summaries/personal/mobile/mobile-plans/Telstra-Upfront-Mobile-Plans-.pdf"

    # response = urllib.request.urlopen(url)
    # pdf_file = BytesIO(response.read())

    pdfReader = PyPDF2.PdfReader(pdf_file)
    number_of_pages = len(pdfReader.pages)
    print("number of page:" + str(number_of_pages) + "\n") 

    text = ""
    for pageNum in range(number_of_pages):
        text += pdfReader.pages[pageNum].extract_text()

    # print(text)

    return text

'''
This function is to call openai API to get the embedding for the provided text
Input: text string
Return: embedding array
'''
def get_openai_embedding(text):
    ## call opanAI to get the embedded value
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text)
    embedding=response['data'][0]["embedding"]

    # print(embedding)
    return embedding

'''
This function will save the provided info to a file with certain format
"filename" | "content" | "embedding"
Delimeter: |
"" double quote: around the cell content

'''
def save_to_file(file, pdfname, content, embedding):
    ## write to a file
    quote="\""
    delimeter="|"
    
    ## for 1st column - file name
    file.write(quote)
    file.write(str(pdfname))
    file.write(quote)
    file.write(delimeter)
    ## for the 2nd column - content
    file.write(quote)
    file.write(text.replace("\n"," "))
    file.write(quote)
    file.write(delimeter)
    ## for the 3rd column - embedding
    file.write(quote)
    file.write(str(embedding))
    file.write(quote)
    ## complete the row
    file.write("\n")


    # # # ## write to a file
    # file.write("\"")
    # file.write(str(embedding))
    # file.write("\"")

    # file.close()

    # # # open the file in the write mode
    # with open('../../data/postpaid_mobile_plan_embedding.csv', 'w') as f:
    #     # create the csv writer
    #     writer = csv.writer(f)

    #     # write a row to the csv file
    #     writer.writerow(str(embedding))
    #     f.close()


##############################
## Main 
##############################

client = connect_minio()

print("\n# Checking if " + minio_bucket_name + " exists\n")
if not client.bucket_exists(minio_bucket_name):
    print("# bucket does not exist, creating new bucket\n")
    client.make_bucket(minio_bucket_name, location="")
else:
    print("# bucket exists\n")

# List all object paths in bucket that begin with prefixname.
objects = client.list_objects(minio_bucket_name, prefix=file_prefix,
                            recursive=True)
    
# open the file in the write mode
file = open(save_file_path,'a')
# to write the header
file.write("filename|content|embedding\n")

for obj in objects:
    print(obj.bucket_name, obj.object_name.encode('utf-8'), obj.last_modified,
        obj.etag, obj.size, obj.content_type)
    
    data = client.get_object(minio_bucket_name, obj.object_name)
    # print(data)

    # print(obj) ## this is the object itself. not the urllib3.response.HTTPResponse 
    # content =  BytesIO(data.read()) ## note: this statement will cause a pdf file save to local

    text = get_pdf_content(BytesIO(data.read()))
    embedding = get_openai_embedding(text)
    # embedding = "[hello world]"

    ## write to the file
    save_to_file(file, obj.object_name, text, embedding)

## finish writing to the local file
file.close()

print("\n# Finish writing to file " + save_file_path + " \n")
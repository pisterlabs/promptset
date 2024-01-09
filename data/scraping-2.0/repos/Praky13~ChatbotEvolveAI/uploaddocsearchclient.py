
import os
import openai
from azure.storage.blob import BlobServiceClient,BlobClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient


from dotenv import load_dotenv

load_dotenv()

# Read the File through the form recognizer module
Form_Recognizer_Key = os.getenv("FR_KEY")
Form_Recognizer_Endpoint = os.getenv("FR_ENDPOINT")

endpoint = Form_Recognizer_Endpoint
key = AzureKeyCredential(Form_Recognizer_Key)



Azure_Search_Service_Name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
Azure_Search_Admin_Key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
Azure_Search_Index_Name = os.getenv("AZURE_SEARCH_INDEX_NAME")



#endpoint = "https://{}.search.windows.net/".format(Azure_Search_Service_Name)

#search_client = SearchClient(endpoint=endpoint,index_name=Azure_Search_Index_Name,credential=AzureKeyCredential(Azure_Search_Admin_Key))

def pdf_text(endpoint,key,pdf_path):
        #  Creating the FormRecognizerClient with a token credential.
        form_recognizer_client = FormRecognizerClient(endpoint, key)
        #Read the pdf file in binary mode
        with open(pdf_path, "rb") as pdf_file:
            pdf = pdf_file.read()
        # Extract all the text from pdf file
        # Extract text and content/layout information from a given document.
        poller = form_recognizer_client.begin_recognize_content(pdf)
        result = poller.result()
        #save the results in list
        extracted_text=[]
        for page in result:
            for line in page.lines:
                extracted_text.append(line.text)
        return(("").join(extracted_text)) 

 #Azure Blob credentials
Azure_Blob_Connection_String = os.getenv("AZURE_BLOB_CONNECTION_STR")
Azure_Blob_Container_Name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

 # Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(Azure_Blob_Connection_String)

# Create a ContainerClient using blob service client
container_client = blob_service_client.get_container_client(Azure_Blob_Container_Name)

blob_list = container_client.list_blobs()

try:

    text = ""
    for blob in blob_list:
        print("\t" + blob.name)
        text+= pdf_text(endpoint,key,blob)

    print(text)




   

   

    

    print("Reached here")
    #result = search_client.upload_documents(documents=text)
    #print("Upload of new document succeeded: {}".format(result[0].succeeded))
except Exception as ex:
    print (ex.message)
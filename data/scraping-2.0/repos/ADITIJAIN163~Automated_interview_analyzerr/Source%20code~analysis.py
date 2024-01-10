from azure.storage.blob import BlobServiceClient
import openai
from fpdf import FPDF
import json
import keys
from openai_helper import complete_openai 

# Connection string for your Azure Storage account

connection_string = keys.connection_string
container_name = keys.container_name
blob_name = "blob.json"
openai.api_type = keys.OpenAI_type
openai.api_base = keys.OpenAI_base
openai.api_version = keys.OpenAI_version
openai.api_key = keys.openAIKey
 
# Setting the format of the pdf generated after evaluation of the candidate
class PDF(FPDF):

        def header(self):

            self.set_font('Arial', 'B', 12)

            self.cell(0, 10, 'Interview Report', align='C', ln=True)

            self.ln(10)

 

        def chapter_title(self, title):

            self.set_font('Arial', 'B', 12)

            self.cell(0, 10, title, 0, 1)

            self.ln(4)

 

        def chapter_body(self, body):

            self.set_font('Arial', '', 12)

            self.multi_cell(0, 10, body)

            self.ln()

 

# function to get data from blob

def GetDataFromBlob():

     # Create a blob service client

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a reference to the container

    container_client = blob_service_client.get_container_client(container_name)

    # Get a reference to the blob

    blob_client = container_client.get_blob_client(blob_name)

    # Download the blob data

    blob_data = blob_client.download_blob().readall()

    return blob_data

 

#function to get Analyze data

def AnalyzedData():

    blob_data=GetDataFromBlob()

    prompt = [ {'role':'system', 'content':"""

You are an professional evaluator. You evaluate answers given by the user against every question asked by the assistant. Give score for each answer out of 100 on the basis of technical correctness of the answer for the question. Give score as 0 if the answer given by user is not correct. Also separately give ratings for communication skills as "excellent", "good" and "bad" based on the answer provided. Give output in the form of collection of lists in which 0th index of list should contain question, 1th index of list should contain answer, 2nd index should contain score and 3rd index should contain communication skill.

                """} ]

    prompt.append({'role':'user','content':f"{blob_data} {' analyze this data'}"})

    result=complete_openai(prompt)
    result1 = json.dumps(result)
    # class start from here

    data=json.loads(result1)
    return data

#function to generate pdf of interview

def pdfGenerator():

    data=AnalyzedData()

    pdf = PDF()

    pdf.add_page()
    sum=0

    for question in eval(data['choices'][0]['message']['content']):

        sum = sum + question[2];
        pdf.chapter_title("Question:")

        pdf.chapter_body(question[0])

        pdf.chapter_title("Answer:")

        pdf.chapter_body(question[1])

        pdf.chapter_title("Techincal score")

        pdf.chapter_body(str(question[2]))

        pdf.chapter_title("Communication Skill:")

        pdf.chapter_body(question[3])

        pdf.ln()
    
    pdf.chapter_title("OverAll Score:")

    pdf.chapter_body(str((sum/500)*100)+"%")

    pdf.ln()

    # Save the PDF to a file

    pdf_file_name ="interview_report"+ ".pdf"

    pdf.output(pdf_file_name)


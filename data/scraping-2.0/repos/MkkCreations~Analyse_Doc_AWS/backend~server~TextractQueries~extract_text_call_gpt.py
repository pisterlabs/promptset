import boto3
import trp
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION_ID")


def get_gpt_answer(data_parsed):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You're a compliance officer in the finance world",
            },
            {
                "role": "user",
                "content": f"Avec le texte ci-dessous, peux-tu me donner les événements importants des ces 3 dernières années ?  {data_parsed} ",
            },
        ],
        temperature=0.3,
    )

    print(response["choices"][0]["message"]["content"])


def get_kv_map(s3BucketName, documentName, diligenceId, documentType):
    client = boto3.client("textract")
    response = client.start_document_text_detection(
        DocumentLocation={
            "S3Object": {
                "Bucket": s3BucketName,
                "Name": str(diligenceId + "/" + documentType + "/" + documentName),
            }
        }
    )

    job_id = response["JobId"]
    response = client.get_document_text_detection(JobId=job_id)
    status = response["JobStatus"]
    while status == "IN_PROGRESS":
        response = client.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

    return response


def uploadS3(s3BucketName, documentName, diligenceId, documentType):
    s3 = boto3.client("s3")
    s3.upload_file(
        Filename=documentName,
        Bucket=s3BucketName,
        Key=str(diligenceId + "/" + documentType + "/" + documentName),
    )


def format_text_detection(textract_json):
    text = ""
    t_doc = trp.Document(textract_json)
    for page in t_doc.pages:
        for line in page.lines:
            text += line.text + " "
    return text


def main():
    s3BucketName = "inputanalyze"
    documentName = "./documents/BNP-HISTORIQUE.png"
    diligenceId = "1"
    documentType = "historique"
    uploadS3(s3BucketName, documentName, diligenceId, documentType)
    response = get_kv_map(s3BucketName, documentName, diligenceId, documentType)
    data_extracted = format_text_detection(response)
    get_gpt_answer(data_extracted)


if __name__ == "__main__":
    main()

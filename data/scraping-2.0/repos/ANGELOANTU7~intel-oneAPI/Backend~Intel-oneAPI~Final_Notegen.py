import os
import boto3
from fastapi import APIRouter
import openai
from intel_extension_pytorch import PyTorchExtension
from intel_extension_tensorflow import TensorFlowExtension
from intel_optimization_xgboost import XGBoostOptimizer
from intel_optimization_modin import ModinOptimizer

app = APIRouter()
s3_access_key = "<your_s3_access_key>"
s3_secret_access_key = "<your_s3_secret_access_key>"
s3_bucket_name = "learnmateai"

s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_access_key)

# Set up OpenAI API credentials
openai.api_key = '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'

# Initialize Intel libraries and tools
pytorch_extension = PyTorchExtension()
tensorflow_extension = TensorFlowExtension()
xgboost_optimizer = XGBoostOptimizer()
modin_optimizer = ModinOptimizer()

@app.get("/note_gen")
async def summarize_s3_files(user:str):
    user=user+"/"
    bucket_name= "learnmateai"
    folder_name= user+"Analysed_Notes"
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
        for file in response['Contents']:
            file_key = file['Key']
            file_name = os.path.basename(file_key)
            print(file_name)
            summary = await summarize_file(bucket_name, file_key,file_name)
            print(summary)
            save_summary(file_name, summary,user)
        return {'message': 'Created Notes and saved successfully.'}
    except Exception as e:
        return {'error': str(e)}

async def summarize_file(bucket_name: str, file_key: str, file_name:str):
    try:
        file_name=file_name.split(".txt")[0]
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        prompt = f'You are a teacher, make a full explanation for the topic: {file_name} below in good format. Include key concepts, explanations, and any relevant information. \nMake sure to cover these topics:\n{file_content}'
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        raise e

def save_summary(file_name: str, summary: str, user):
    try:
        file_name = file_name.split(".txt")[0]
        save_key = f'{user}Notes_Topicwise/{file_name}.txt'
        s3.put_object(Body=summary, Bucket=s3_bucket_name, Key=save_key)
    except Exception as e:
        raise e

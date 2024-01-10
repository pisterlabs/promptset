import os
import boto3
from fastapi import APIRouter
import openai
import json

app = APIRouter()
s3_access_key = ""
s3_secret_access_key = ""
s3_bucket_name = "learnmateai"

s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_access_key)

# Set up OpenAI API credentials
openai.api_key = ''

def processor( data,current_data, final_date):

    prompt = f'make a study plan of these topics:\n{data} \n\n current date:{current_data} \n final date:{final_date} \n\n the output should strictly be a json array of (Topic_name,date) , the date should be perfectly decided so that student can learn easily between the current date and final date, the repeat the topics based on their difficulty \n \n output should not contain anything expect the json, and it should not exceed 4000 token limit'
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

@app.post("/genStudyPlan")
async def generateStudyPlan(email: str,current_date: str,final_date: str):
    text=""
    bucket_name= "learnmateai"
    folder_name= f"{email}/Analysed_Notes"
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
        for file in response['Contents']:
            file_key = file['Key']
            print(file_key)
            file_name = os.path.basename(file_key)
            file_name=file_name.split(".txt")[0]
            text=text+"\n"+file_name
            
            
        print(text)
        
        json_text=processor(text,current_date,final_date)    
        save_plan(email,json_text)

        # Return the JSON content
        return {"status" : "plan created successfully"}
    except Exception as e:
        return {'error': str(e)}




def save_plan(email,summary):
    try:
        save_key = f'{email}/StudyPlan/studyplan.json'
        s3.put_object(Body=summary, Bucket=s3_bucket_name, Key=save_key)
    except Exception as e:
        raise e


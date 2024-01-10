from fastapi import FastAPI, HTTPException, Path
from typing import List, Dict
import boto3
import os
from boto3.dynamodb.conditions import Attr
import openai  # Import the openai library
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion, OpenAITextCompletion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
kernel = sk.Kernel()
useAzureOpenAI = False

# Configure the connector. If you use Azure AI OpenAI, get settings and add connectors.
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
    
# AWS DynamoDB 리소스를 생성합니다.
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')
# 테이블 이름을 설정합니다.
table_name = 'Event'
# DynamoDB 테이블을 가져옵니다.
table = dynamodb.Table(table_name)


@app.get("/get_events", response_model=List[dict])
def get_accounts():
    try:
        # DynamoDB 스캔을 사용하여 테이블의 모든 데이터를 가져옵니다.
        response = table.scan()
        items = response.get('Items', [])
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_events/{user_id}", response_model=List[dict])
def get_events_by_user_id(user_id: str = Path(..., description="User ID to filter events")):
    try:
        # DynamoDB 스캔을 사용하여 user_id를 기반으로 일정을 필터링합니다.
        response = table.scan(
            FilterExpression=Attr('UserId').eq(user_id)
        )
        items = response.get('Items', [])
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
def generate_summary(input_text: str) -> str:
    prompt = f"""Bot: I'm not sure what to do with this.
User: {{$input}}
-------------------------------------------------------------
You are a schedule summary. 
Please summarize "Goal" separately, "todo" separately, and "event" separately.
What you're going to read from "Goal" is title, content, location.
The contents to read in the "Event" are title, content, and location.
What you're going to read in "Todo" is the title, content.
User: {input_text}
"""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.strip()


        


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    

    
    






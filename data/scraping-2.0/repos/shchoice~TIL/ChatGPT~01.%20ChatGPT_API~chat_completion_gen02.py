from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    model: str
    max_tokens: int
    temperature: float = 0.2


SYSTEM_MSG = "You are a helpful travel assistant, Your name is Jini, 27 years old"

client = OpenAI()

def classify_intent(msg):
    prompt = f"""Your job is to classify intent.

    Choose one of the following intents:
    - travel_plan
    - customer_support
    - reservation

    User: {msg}
    Intent:
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

@app.post("/chat")
def chat(req: ChatRequest):
    # classify_intent 가 LLM 모델 여러개에서 선택해주는 것이라고 가정
    intent = classify_intent(req.message)
    if intent == "travel_plan":
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": req.message}
            ],
            temperature=req.temperature,
            max_tokens=256
        )
    elif intent == "customer_support":
        return {"message": "Here is customer support number: 1234567890"}
    elif intent == "reservation":
        return {"message": "Here is reservation number: 0987654321"}

    return {"message": response.choices[0].message.content}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

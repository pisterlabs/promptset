import os
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from starlette import status

# Set up agent service auth - needed to register with Agent Wallet
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "call_me_with_this_key")
security = HTTPBearer()

# Middleware that authenticates incoming API calls
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
  if credentials.credentials != AGENT_API_KEY:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid API Key")

# Set up agent API service
app = FastAPI(dependencies=[Depends(verify_api_key)])

# Agent configuration using LangChain
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

# Using LangServe adds routes to the app:
# /chat/invoke
# /chat/batch
# /chat/stream
add_routes(app, prompt | model, path="/chat")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="localhost", port=8000)
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import logging
from langchain.llms import HuggingFaceHub 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_model():
    try:
        load_dotenv()
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        # Initialize HF LLM
        flan_t5 = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 256}
        )

        # Build prompt template for simple question answering
        template = """ Question: {question}

        Answer: """

        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm_chain = LLMChain(
            prompt=prompt,
            llm=flan_t5
        )

        return llm_chain
    except Exception as e:
        logging.error(f"Error initializing model: {e}", exc_info=True)
        raise  # Re-raise the exception to propagate it further

def setup_fastapi(llm_chain):
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def read_root():
        with open('templates/chatbot.html', 'r') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)

    @app.get("/chatbot/{user_message}")
    def get_bot_response(user_message: str):
        try:
            # Process the user's message and get the bot's response
            bot_response = process_message(llm_chain, user_message)
            return {"answer": bot_response}
        except Exception as e:
            logging.error(f"Error processing message: {e}", exc_info=True)
            return {"answer": "Sorry, something went wrong."}

    return app

def process_message(llm_chain, user_message):
    try:
        question = [{"question": user_message}]
        response = llm_chain.generate(question)
        return response.generations[0][0].text
    except Exception as e:
        logging.error(f"Error generating response: {e}", exc_info=True)
        return "Sorry, I couldn't understand your message."

if __name__ == "__main__":
    try:
        llm_chain = initialize_model()
        fastapi_app = setup_fastapi(llm_chain)
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"Error during initialization: {e}", exc_info=True)

import os
from langchain.document_loaders import PyPDFLoader
import re
from uagents.setup import fund_agent_if_low
from uagents import Agent, Bureau, Context, Model
from Messages.List_Modal import Message
import time
import json

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


Converter = Agent(
    name = "Converter",
    port = 8000,
    seed = "Pushpak_Agrawal",
    endpoint=["http://127.0.0.1:8000/b"],
)

fund_agent_if_low(Converter.wallet.address())
file_path = 'Jobs.json'

@Converter.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    for resume in msg.message:
         with open(file_path, 'r') as file:
            Job_Queries = json.load(file)
            for key, value in Job_Queries.items():
                os.environ["OPENAI_API_KEY"] = "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" # this API key has been deactivated. Generate a new key at openai.com

                pdf_loader = PyPDFLoader(resume)
                documents = pdf_loader.load()

            
                chain = load_qa_chain(llm=OpenAI())
                query = f"""
                Rate this person's compatability for a job
                Make sure that the person meets the minimum educational qualifications for the job and is highly skilled.
                Rate strictlty according to educational qualifications 
                Be Extremely Critical and try to give as less of a score as possible consider worst case scenario
                See the resume from a recruiters point of view
                from 0 to 1000

                Make sure it matches the following job description:
                {value}
                If cannot accurately tell the answer just guess but make sure that the response should be a number between 0 and 1000
                """
                score = None
                response = chain.run(input_documents=documents, question=query)
                score = [ word for word in response.split(" ") if word.isdigit() and int(word) < 1000]
                
                temp_resume= ""
                if score:
                    temp_resume = resume[18:]
                    temp_resume = temp_resume[:-4]

                    ctx.logger.info(f"Resume {temp_resume} got: {max(score)} for job: {key}")

                time.sleep(20)


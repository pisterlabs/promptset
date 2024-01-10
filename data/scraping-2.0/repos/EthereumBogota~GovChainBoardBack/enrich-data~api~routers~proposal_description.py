'''
Include all the routes starting at /event/
'''
from fastapi import APIRouter, Depends, Query

from sqlmodel import Session, select
from sqlalchemy import desc

from typing import List
from api.services.database import get_session
from api.models.models import IndexDataBasin

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
import os 
import openai

import w3storage
import json
import uuid





router = APIRouter(
    prefix="/proposal_description",
    tags=["proposal_description"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)


@router.post("/")
async def enrich_proposal_description(*, contract_id: str, proposals: dict[str, str],  db: Session = Depends(get_session)):
    '''
    proposals: dict[str, str] = [(proposalId, description)]
    TODO: Test
    '''
    # openai.organization = "org-denUIc5wfgQ2ruU7O5x1OozD"


    task = '''
    You are a classifier of proposals from a DAO, clasify the proposal in one word from 
    [
        "Governance",
        "Funding",
        "Projects",
        "Smart Contracts",
        "Tokens",
        "Partnerships",
        "Community Engagement",
        "Technology",
        "Research and Development",
        "Security",
        "Legal and Compliance",
        "Tokenomics",
        "Environmental Impact",
        "Economic Incentives",
        "DAO Treasury Management",
        "Governance Token Distribution",
        "Education and Awareness",
        "Charitable and Social Impact",
        "Protocol Upgrades",
        "Strategic Direction"
    ]
    . Give only a json list ef máximum 5 elements of the words that correspond. Nothing else, if is empty then [], do not give hints or any other indication.
    only return a List not a Dict or any other data type. Do not ask for more information. If you do not know the return []. 

    '''

    openai.api_key = os.environ["OPEN_API_KEY"]
    categories = {}
    for proposalId, description in proposals.items():
        response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": task},
            {"role": "user", "content": description},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            # {"role": "user", "content": "Where was it played?"}
            ]
        )
        categories[proposalId] = response['choices'][0]['message']['content']

    
    resultado =  { "contract_id": contract_id, "values": categories}

    # save resultado on Web3.Storage
    # TODO:  save resultado on Web3.Storage
    # Connect to web3_storage
    w3 = w3storage.API(token=os.environ["WEB3STORAGE_KEY"])
    # some_uploads = w3.user_uploads(size=25)

    cid = w3.post_upload(( 'enrich_proposal_description3.json',json.dumps(resultado, indent = 4) ))
    print(cid)
    # Save Cid y Id of the contract on DB and Datetime
    event = IndexDataBasin(personalized_id=uuid.uuid4(),cid=cid, contract_id=contract_id)
    db.add(event)
    db.commit()

    return resultado # Or CID


@router.get("/contract_data_cid")
async def get_cid(*,contract_id: str,  db: Session = Depends(get_session)):
    latest_event = db.query(IndexDataBasin).filter(IndexDataBasin.contract_id == contract_id).order_by(desc(IndexDataBasin.created_at)).first()
    
    if(not latest_event):
        return {"status": "failed"}
    return latest_event

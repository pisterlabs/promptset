from fastapi import APIRouter , Request , Form , HTTPException
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.db.base import UniqueConstraintError
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Optional
from fastapi.responses import JSONResponse
import requests
from datetime import datetime
from transformers import BertTokenizer
import json



model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

client = chromadb.Client()


from datetime import datetime

def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError("Type not serializable")

def encode_field(value):
    try:
        # 텍스트 데이터 또는 날짜 데이터인 경우에만 임베딩 수행
        if isinstance(value, str):
            return model.encode(value)
        elif isinstance(value, datetime):
            # ISO 8601 형식의 문자열로 변환
            iso_format = value.isoformat()
            return model.encode(iso_format)
        elif isinstance(value, list):
            # 리스트인 경우 각 항목에 대해 임베딩 수행
            return [model.encode(str(item)) for item in value]
        elif isinstance(value, dict):
            # 딕셔너리인 경우 각 값에 대해 임베딩 수행
            return {key: model.encode(str(val)) for key, val in value.items()}
        else:
            return value  # None인 경우 그대로 반환
    except Exception as e:
        print(f"Error encoding field: {value}, {e}")
        raise ValueError(f"Error encoding field: {value}, {e}")



client = chromadb.PersistentClient()

openai.api_key = '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'

POrouter = APIRouter(prefix="/posting")

class WorkTypeDTO(BaseModel):
    workCode: int
    workConditions: str
    postingCode: int

class SkillDTO(BaseModel):
    skillCode: int
    skillName: str
    postingCode: int

class PostingExperienceDTO(BaseModel):
    experienceCode: int
    experienceLevel: str
    postingCode: int

class CompanyDTO(BaseModel):
    companyId: int
    email: Optional[str]
    password: Optional[str]
    phoneNumber: Optional[str]
    company: str
    companyType: str
    employeesNumber: int
    establishmentDate: datetime
    companyHomepage: Optional[str]

class PostingDTO(BaseModel):
    postingCode: int
    postingDate: str
    endDate: str
    education: str
    viewCount: int
    location: str
    position: str
    closingForm: str
    content: str
    postingTitle: str 
    selectedCareer: Optional[str]
    selectedConditions: Optional[str]
    selectedSkills: Optional[str]
    workTypeList: List[WorkTypeDTO]
    skillList: List[SkillDTO]
    postingExperienceList: List[PostingExperienceDTO]
    company: CompanyDTO


@POrouter.post("/regist")
async def registCompany(posting: PostingDTO):
    postingCode = posting.postingCode

    company = posting.company

    postingData = [posting.postingTitle , posting.postingDate , posting.endDate, posting.education,
                    company.email , company.company, company.companyType, str(company.employeesNumber),
                    company.establishmentDate]
    


    for work_type in posting.workTypeList:
        
        postingData.append(work_type.workConditions)


    for skill in posting.skillList:
        
        postingData.append(skill.skillName)

    
    for exp in posting.postingExperienceList:
        
        postingData.append(exp.experienceLevel)


    merged_string = " ".join(map(str, postingData))

    embeddings = model.encode(merged_string)

    
    # ChromaDB에 데이터 저장
    collection_name = "posting"

    collection = client.get_collection(name=collection_name)

    
    
    data = {
        "embeddings": [embeddings.tolist()],
        "documents": [merged_string],
        "ids": [str(postingCode)],
    }
    
    
    collection.add(**data)

    print("됐냐?")

    return "gd"

@POrouter.get("/get")
async def get_posting():

    
    try:
        # ChromaDB에서 조회
        collection_name = "posting"
        collection = client.get_collection(name=collection_name)

        query_text = "fuck"
        query_embedding = model.encode(query_text)

        result = collection.query(
            # query_texts=[model.encode("spring")],
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )


        if result:
            # 결과 반환
            print(result)
            return JSONResponse(content=result, status_code=200)
        else:
            raise HTTPException(status_code=404, detail="Posting not found")
    except Exception as e:
        print(f"Error in get_posting: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")




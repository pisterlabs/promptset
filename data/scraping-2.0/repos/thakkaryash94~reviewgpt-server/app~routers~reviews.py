import re
import subprocess
from urllib.parse import urlparse, urlunparse
from fastapi import APIRouter, Depends, Request, status

from pydantic import BaseModel, Field

from typing import Annotated, Any
from sqlalchemy.orm import Session

# from langchain.llms import Ollama
# from langchain.embeddings import OllamaEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OllamaEmbeddings
# from langchain.prompts.prompt import PromptTemplate
# from langchain.vectorstores import Chroma
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from langchain.chains import RetrievalQA
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chat_models import ChatOllama
# from langchain.schema import HumanMessage
# from sqlalchemy.orm import Session

from app.chatmodel import generate_one_time_answer
from app.constants import MODEL, RECAPTCHA_SITEVERIFY_URL, RECAPTCHA_TOKEN
from app.database import crud, models, schemas
from app.database.database import get_db
from app.ipdetails import get_ip_details
from app.logger import get_logger
from app.recaptcha import recaptcha_verify

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewBody(BaseModel):
    url: str = Field(
        examples=[
            "https://www.amazon.in/boAt-Airdopes-161-Playtime-Immersive/product-reviews/B09N7KCNL6",
            "https://www.flipkart.com/boat-airdopes-161-40-hours-playback-asap-charge-10mm-drivers-bluetooth-headset/product-reviews/itm8a7493150ae4a?pid=ACCG6DS7WDJHGWSH&lid=LSTACCG6DS7WDJHGWSH4INU8G&marketplace=FLIPKART",
        ],
    )
    token: str = Field(examples=["Recaptcha Token"])


dbDep: Session = Annotated[dict, Depends(get_db)]

logger = get_logger("reviews")


# @router.post("/one-time", response_model=schemas.ReviewResponse)
@router.post("/one-time")
async def get_one_time_review(request: Request, body: ReviewBody, db: dbDep) -> Any:
    logger.info("Request started")
    url = body.url
    product_id: str
    if re.search("/dp/", url):
        product_id = re.search(r"dp/(.+)/", url).group(1)
    if re.search("/product-reviews/", url):
        product_id = re.search(r"product-reviews/(.+)/", url).group(1)
    parsed_url = urlparse(url)
    website = urlunparse((parsed_url.scheme, parsed_url.netloc, "", "", "", ""))
    url = f"{website}/product-reviews/{product_id}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=avp_only_reviews"

    client_ip = request.client.host
    payload = {"secret": RECAPTCHA_TOKEN, "response": body.token}
    recaptcha_result = recaptcha_verify(payload)
    count = crud.count_othistory_by_ip(db=db, ip_address=client_ip)
    ip_info = get_ip_details(client_ip)
    history = schemas.OTHistoryCreate(
        url=url,
        ip_address=client_ip,
        ip_info=ip_info,
        status=models.OTHistoryEnum.PENDING,
    )
    dbHistory = crud.create_othistory(db=db, item=history)
    # if not recaptcha_result.get("success"):
    #     dbHistory.status = models.OTHistoryEnum.REJECTED
    #     dbHistory = crud.update_othistory(db=db, item=dbHistory)
    #     return {
    #         "code": status.HTTP_401_UNAUTHORIZED,
    #         "message": "Unauthorized Access",
    #         "success": True,
    #     }
    # if count >= 5:
    #     dbHistory.status = models.OTHistoryEnum.REJECTED
    #     dbHistory = crud.update_othistory(db=db, item=dbHistory)
    #     return {
    #         "code": status.HTTP_429_TOO_MANY_REQUESTS,
    #         "message": "Too many requests from same IP",
    #         "success": False,
    #     }
    logger.info(f"Scraping {url}")
    result: Any
    crawler = "amazon"
    if "https://www.flipkart" in url:
        crawler = "flipkart"
    result = subprocess.run(
        [
            "scrapy",
            "crawl",
            f"{crawler}",
            "-a",
            f"url={url}",
            "-a",
            f"page=1",
        ],
        capture_output=True,
        timeout=20,
        text=True,
    )
    logger.info("Reviews fetched successfully")
    reviews = result.stdout
    if reviews == "":
        dbHistory.status = models.OTHistoryEnum.FAILED
        dbHistory = crud.update_othistory(db=db, item=dbHistory)
        return {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Something went wrong",
            "success": False,
        }
    data = generate_one_time_answer(reviews)
    if result.stdout:
        dbHistory.status = models.OTHistoryEnum.SUCCESS
        dbHistory = crud.update_othistory(db=db, item=dbHistory)
        return {
            "code": status.HTTP_200_OK,
            "message": "Response retrieved successfully",
            "data": data,
            "success": True,
        }
    else:
        dbHistory.status = models.OTHistoryEnum.FAILED
        dbHistory = crud.update_othistory(db=db, item=dbHistory)
        return {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Something went wrong",
            "success": False,
        }


class QuestionBody(BaseModel):
    question: str


# @router.post("/reviews/question")
# def post_question(body: QuestionBody, db: Session = Depends(get_db)):
#     ollama = Ollama(base_url="http://localhost:11434", model=MODEL)
#     oembed = OllamaEmbeddings(base_url="http://localhost:11434", model=MODEL)
#     client = chromadb.HttpClient(host="127.0.0.1", port=8000)
#     crud.create_history(db=db, history="")

#     vectorstore = Chroma(
#         client=client,
#         collection_name="amz_reviews",
#         embedding_function=oembed,
#     )
#     documents = vectorstore.get().get("documents")

#     # Prompt
#     # template = """Use the following pieces of context to answer the question at the end.
#     # If you don't know the answer, just say that you don't know, don't try to make up an answer.
#     # Use three sentences maximum and keep the answer as concise as possible.
#     # {context}
#     # Question: {question}
#     # Helpful Answer:"""
#     # QA_CHAIN_PROMPT = PromptTemplate(
#     #     input_variables=["context", "question"],
#     #     template=template,
#     # )
#     # qachain = RetrievalQA.from_chain_type(
#     #     llm=ollama,
#     #     retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
#     #     chain_type="stuff",
#     #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#     #     return_source_documents=True,
#     # )
#     # qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
#     # result = qachain({"query": body.question})
#     # return result
#     chat_model = ChatOllama(
#         model=MODEL,
#         format="json",
#         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     )
#     messages = [
#         HumanMessage(content="Below are the reviews of the product. Analyze them"),
#         HumanMessage(content="\n".join(documents)),
#         HumanMessage(content=body.question),
#     ]
#     print("\n".join(documents))
#     chat_model_response = chat_model(messages)
#     return chat_model_response

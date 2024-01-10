from django.shortcuts import render
import time
from rest_framework.views import APIView
# from rest_framework.renderers import JSONRenderer
# from rest_framework.response import Response
from django.views.generic import View  
from django.shortcuts import render
from django.http import StreamingHttpResponse
from rest_framework import status
from openai import OpenAI  # for OpenAI API calls
import time  # for measuring time duration of API calls
import os
from .serializers import ImageSerializer
from django.conf import settings
#--------------------------------------------------
# AI PART
#--------------------------------------------------
from dotenv import load_dotenv

import cohere
import re
from langchain.vectorstores import AstraDB
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from .models import image
import google.generativeai as genai
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)
from PIL import Image
from django.conf import settings
import google.generativeai as genai

from rest_framework.response import Response


ASTRA_DB_APPLICATION_TOKEN = settings.ASTRA_DB_APPLICATION_TOKEN
ASTRA_DB_ID = settings.ASTRA_DB_ID
ASTRA_DB_API_ENDPOINT = settings.ASTRA_DB_API_ENDPOINT
COHERE_API_KEY = settings.COHERE_API_KEY

# Google client
GOOGLE_API_KEY = settings.GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Cohere client
COHERE_API_KEY = settings.COHERE_API_KEY
co = cohere.Client(COHERE_API_KEY)
# Vector DB already loaded.(Already indexed)
# Get the vectors from the DB (using Retriever)


def split_and_convert(string):

    pattern = r'\([^)]+\)\s*>\s*'
    components = re.split(pattern, string)
    components = [c.strip() for c in components if c.strip()]
    return [{"text": component} for component in components]


def get_retriever(question):

    cohereEmbedModel = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

    astra_vector_store = AstraDB(
        embedding=cohereEmbedModel,
        collection_name="tailwind_docs_embeddings",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        # batch_size = 20,
    )

    # astraOP = astra_vector_store.as_retriever(search_kwargs={'k': 10})
    return rerank_documents(question,astra_vector_store.as_retriever(search_kwargs={'k': 10}))
    
    # return astra_vector_store.as_retriever(search_kwargs={'k': 2})


def rerank_documents(question,astraOP):

    # print(astraOP.get_relevant_documents("What is Tailwind ?"))
    retrieved_documents = astraOP.get_relevant_documents(question)
    # print(type(retrieved_documents[0]))
    # Convert the retrieved documents to the required format for rerank
    rerank_documents = [{"id": str(i), "text": doc} for i, doc in enumerate(retrieved_documents)]
    # print(rerank_documents)
    # print(type([rerank_documents]))
    docs = []
    for i in rerank_documents:
        docs.append(i['text'].page_content)

    results = co.rerank(query=question, documents=docs, top_n=3, model="rerank-multilingual-v2.0")
    # print(results)
    return results


def get_prompt_template():
    """ Prompt Template from Langchain
        with context and question. Can change to give dfferent format output
    """

    template = """You are an experienced senior TAILWIND-CSS developer, recognized for my efficiency in building websites and converting plain CSS into TAILWIND-CSS code. 
    Your expertise extends to NEXT JS development as well. You are eager to assist anyone with any TAILWIND-CSS related inquiries or tasks others may have.
    The context to answer the questions:
    {context}
    Return you answer as markdown.
    And if you only receive CSS as input, first create a simple HTML template and then convert that CSS into TAILWIND-CSS with reference to the relevant docs from above context 
    and add it in HTML. Only give HTML with TAILWIND-CSS as reply.

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt


def gemini_Chain(question):

    Gemini_llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True)

    """RETRIEVER WORKING - ✅"""
    # print(astra_retriever.get_relevant_documents("what is Tailwind?"))
    # astra_retriever.get_relevant_documents("what is Tailwind?")

    """RERANK THE RETREIVED DOCUEMENTS WORKING - ✅"""
    # use cohere reranker

    """PROMPT WORKING - ✅"""
    gemini_prompt = get_prompt_template()

    """OUTPUT PARSER WORKING - ✅"""
    output_parser = StrOutputParser()

    """If reranker working add the OP of rereanker in context.
        As of now it is the top n from relevant documnets.
    """
    # print(reranked)
    chain = RunnableMap({
        "context": lambda x: get_retriever(x["question"]),
        "question": lambda x: x["question"]
    }) | gemini_prompt | Gemini_llm | output_parser

    res = chain.stream({"question": question})
    for r in res:
        yield r

#--------------------------------------------------
        # Gemini CSS to Tailwind Converter
#--------------------------------------------------

GOOGLE_API_KEY = settings.GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)



class GeminiAPI(APIView):
    # Vector DB already loaded.(Already indexed)
    # Get the vectors from the DB (using Retriever)
    def main_test():

        """To run the main code run : gemini_Chain()"""
        gemini_Chain(question="What is Tailwind ?")

        """To check if reranker is wokriing run get_retriever"""
        # get_retriever()
    
    def post(self,request):
        print(request.data)
        if request.data['message'] == '':
            chat = gemini_Chain(question='Send a greetings message for me and ask me to ask you a question to continue a conversation')
        else:
            chat = gemini_Chain(question=request.data['message'])
        response =  StreamingHttpResponse(chat,status=200, content_type='text/event-stream')
        return response

#--------------------------------------------------
    #Next Tail LLM
#--------------------------------------------------

def next_Tail_llm_prompt_template():
    """ Prompt Template from Langchain
        with context and question. Can change to give dfferent format output
    """

    template = """You are an experienced senior TAILWIND-CSS developer, recognized for my efficiency in building websites and converting plain CSS into TAILWIND-CSS code. 
    Your expertise extends to NEXT JS development as well. You are eager to assist anyone with any TAILWIND-CSS related inquiries or tasks others may have.
    The context to answer the questions:
    {context}

    Question: {question}
    if the question is a greeting message, reply with a greeting message and ask me to ask you a question related to Nextjs and Tailwind to continue a conversation. Don't any introduction about me in the greeting message
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt

def next_Tail_llm(question):

    Gemini_llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True)

    """RETRIEVER WORKING - ✅"""
    # print(astra_retriever.get_relevant_documents("what is Tailwind?"))
    # astra_retriever.get_relevant_documents("what is Tailwind?")

    """RERANK THE RETREIVED DOCUEMENTS WORKING - ✅"""
    # use cohere reranker

    """PROMPT WORKING - ✅"""
    gemini_prompt = next_Tail_llm_prompt_template()

    """OUTPUT PARSER WORKING - ✅"""
    output_parser = StrOutputParser()

    """If reranker working add the OP of rereanker in context.
        As of now it is the top n from relevant documnets.
    """
    # print(reranked)
    chain = RunnableMap({
        "context": lambda x: get_retriever(x["question"]),
        "question": lambda x: x["question"]
    }) | gemini_prompt | Gemini_llm | output_parser

    res = chain.stream({"question": question})
    for r in res:
        yield r


class NextTailLLM(APIView):

    def post(self,request):
        print(request.data)
        if request.data['message'] == '':
            chat = next_Tail_llm(question='Send a greetings message for me and ask me to ask you a question to continue a conversation')
        else:
            chat = next_Tail_llm(question=request.data['message'])
        response =  StreamingHttpResponse(chat,status=200, content_type='text/event-stream')
        return response

#--------------------------------------------------
        # UI to Code
#--------------------------------------------------
from io import BytesIO
import requests

def first ():
    image_urls = [
        "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
        "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2Fchatgpt_ea8cd42076_83f28f8b57.png&w=640&q=75",
        "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2Fsocio_AI_abae5695af_dc9fd39a38.png&w=640&q=75",
        "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2FAihika_634ed5a8ee.png&w=640&q=75",
        "https://www.softservedweb.com/_next/image?url=http%3A%2F%2F128.199.22.214%2Fuploads%2Fcult_88100bc876_e7509e2d58.png&w=640&q=75",
        # Add yours here!
    ]

    image_documents = load_image_urls(image_urls)
    gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")
    # import matplotlib.pyplot as plt

    img_response = requests.get(image_urls[0])
    print(image_urls[0])
    img = Image.open(BytesIO(img_response.content))
    # plt.imshow(img)

    stream_complete_response = gemini_pro.stream_complete(
        prompt="Give me more context for the images",
        image_documents=image_documents,
    )

    for r in stream_complete_response:
        print(r.text, end="")


from llama_index.multi_modal_llms.gemini import GeminiMultiModal

from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)

from trulens_eval import TruCustomApp
from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
from trulens_eval import Provider
from trulens_eval import Feedback
from trulens_eval import Select
from trulens_eval import TruCustomApp
tru = Tru()
tru.reset_database()
gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")

# create a custom class to instrument
class Gemini:
    @instrument
    def complete(self, prompt, image_documents):
        completion = gemini_pro.complete(
            prompt=prompt,
            image_documents=image_documents,
        )
        return completion

# create a custom gemini feedback provider
class Gemini_Provider(Provider):
    def UI_rating(self, image_url) -> float:
        image_documents = load_image_urls([image_url])
        city_score = float(gemini_pro.complete(prompt = "Is this image of a UI? Respond with the float likelihood from 0.0 (not UI) to 1.0 (UI).",
        image_documents=image_documents).text)
        return city_score

gemini_provider = Gemini_Provider()
f_custom_function = Feedback(gemini_provider.UI_rating, name = "UI Understandability").on(Select.Record.calls[0].args.image_documents[0].image_url)

def ui_to_code(url,prompt="Convert this image into HTML and TAILWIND CSS code") :

    image_urls = [
        url
        # Add yours here!
    ]
    print("prompt: ",prompt)
    image_documents = load_image_urls(image_urls)
    gemini = Gemini()

    # gemini_provider.city_rating(image_url=url)
    # tru_gemini = TruCustomApp(gemini, app_id = "gemini", feedbacks = [f_custom_function])

    # with tru_gemini as recording:
    res = gemini.complete(
    prompt=prompt,
    image_documents=image_documents
    )

    
    return res
        # tru.run_dashboard()

class UItoCode(APIView):

    def post(self,request):
        print("hello")
        serializer = ImageSerializer(data=request.data)
        image_urls = []
        
        if serializer.is_valid():
            # Save the image to the model
            print(serializer.validated_data)
            saved_image = serializer.save()
            image_urls.append(saved_image.image.url)
            print(image_urls)

            # image_urls.append('https://blr1.digitaloceanspaces.com/next-tail-space/next-tail/images/website-232.jpg?AWSAccessKeyId=DO00DRTKKFNWF7Z4YT8T&Signature=Pgq1seOVRm8WUNjVRoz9%2BqaFQ1M%3D&Expires=1703442581')
            prompt="Convert this image into HTML using TAILWIND CSS code as cdn"
            # print("prompt: ",prompt)
            image_documents = load_image_urls(image_urls)
            gemini = Gemini()

            
            gemini_provider.UI_rating(image_url=image_urls[0])
            tru_gemini = TruCustomApp(gemini, app_id = "gemini", feedbacks = [f_custom_function])

            with tru_gemini as recording:

                """Return or yield this 'res'
                """

             
                # print(recording)
                # print("TRU GEMINI >>>>>>>> \n", tru_gemini)
                print(image_urls)
                if serializer.validated_data['prompt'] == '':
                    chat = gemini.complete(
                    prompt=prompt,
                    image_documents=image_documents
                    )
                else:
                    chat = gemini.complete(
                    prompt=prompt+" & "+serializer.validated_data['prompt'],
                    image_documents=image_documents
                    )
                print(chat)

            print("TrueLens Eval:\n------------\n",tru.get_leaderboard(app_ids = ["gemini"]))
            response =  Response(chat,status=200)
        else:
            response = Response("Invalid data", status=status.HTTP_200_OK)
        return response

# -------------------------------------------------- 
    # View Class
# --------------------------------------------------    
class HomeView(View):

    def get(self,request):
        return render(request,'index.html')
    
class cssToTailwindView(View):

    def get(self,request):
        return render(request,'tailwind_conversion.html')
    
class nextAIView(View):

    def get(self,request):
        return render(request,'next.html')
    
class UItoCodeView(View):

    def get(self,request):
        return render(request,'ui_to_code.html')

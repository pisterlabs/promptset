import io
import os
import nlpcloud
import re
from fastapi import UploadFile, status, HTTPException
from PyPDF2 import PdfReader

from langchain.chat_models import ChatOpenAI
import json
import tiktoken

from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback

from src.database.repository.contents_repository import ContentsRepository
from src.service.response_question_service import ResponseQuestionService
from src.entity.contents import ContentsEntity, ConsultContentInformation, UpdateContents

from src.utils.environment.env import setting
from src.database.repository.fruit_fungal_diseases_repository import FruitFungalDiaseasesRepository

contents_repository = ContentsRepository()
response_question_service = ResponseQuestionService()
fruits_fungal_disease_repository = FruitFungalDiaseasesRepository()


class ContentsService:

    def extract_array(self, text: str):

        pattern = r'\[.*?\]'

        matches = re.findall(pattern, text)
        if len(matches) > 0:
            first_match = matches[0]

            try:
                result = eval(first_match)
                return result
            except SyntaxError:
                return []
        else:
            return []

    async def _extract_text_from_pdf(self, pdf_file: UploadFile):
        pdf_content = await pdf_file.read()

        pdf_reader = PdfReader(io.BytesIO(pdf_content))

        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        return pdf_text

    async def _splitter_data(self, file):
        text = await self._extract_text_from_pdf(file)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        information = text_splitter.split_text(text)
        return information

    async def create_contents(self, user_id: str, data_contents: ContentsEntity, file: UploadFile):
        information = await self._splitter_data(file)
        document_contents = {
            "information": information,
            "user_id": user_id,
            "title": file.filename.replace(".pdf", ""),
            "category": data_contents.category,
            "fruit": data_contents.fruit
        }
        content = contents_repository.create_contents(document_contents)
        return content

    def search_info_embedding_by_nlCloud(self, user_id, search: ConsultContentInformation):

        contents_db = contents_repository.search_info_contents(user_id, search)

        client = nlpcloud.Client("finetuned-llama-2-70b", "5bb1ec4f6a2e9d62027048f1c879eefdee2a5f51", gpu=True)

        response = client.question(

            question="""Como experto en biología y enfermedades de plantas, por favor, clasifica las dos enfermedades de plantas más comunes que afectan a las. Proporciona sus nombres exactos en un arreglo como este: ['Enfermedad 1', 'Enfermedad 2']. Limita tu respuesta solo a las enfermedades que te proporcione en el contexto. 
             ¿{}?.""".format(search.question),
            context=contents_db["information"]
        )


        print(response)
        diseases = self.extract_array(response["answer"])


        response_question = {
            "user_id": user_id,
            "response": response["answer"],
            "fruit": search.fruit,
            "prompt": search.question,
            "answer_correct": True
        }
        response_question_service.create_response_question(response_question)


        consult = {
            'scientific_name': {
                '$in': [re.compile(re.escape(disease), re.IGNORECASE) for disease in diseases]
            }
        }

        result = fruits_fungal_disease_repository.find_fruit_disease_by_consult(consult)


        if result is None:
            return []
        serialized_response_fruit = []
        for result_fruit in result:
            result_fruit['_id'] = str(result_fruit['_id'])
            serialized_response_fruit.append(result_fruit)
        return serialized_response_fruit

        return {"fruit_disease": result}

    def search_info_embedding_by_OpeIA(self, user_id, search: ConsultContentInformation):
        os.environ['OPENAI_API_KEY'] = setting.OPEN_KEY

        contents_db = contents_repository.search_info_contents(user_id, search)
        if contents_db is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The collection was not found"
            )
        ##Guardar en documento el search
        contents = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(contents_db["information"], contents)
        docs = knowledge_base.similarity_search(search.question)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs,
                                 question=f"Como experto en biología y enfermedades de plantas, por favor, clasifica las dos enfermedades de plantas más comunes que afectan a las. Proporciona sus nombres exactos un array de esta manera: ['Enfermedad 1', 'Enfermedad 2'] nada mas!. Limita tu respuesta solo a las enfermedades que te proporcione en el contexto. ; ${search.question}, Damela en un array y solo los nombres nada mas, de esta manera: ['enfermedad1', 'enfermedad2' ]!! ")


        response_question_gtp = {
            "user_id": user_id,
            "response": response,
            "fruit": search.fruit,
            "prompt": search.question,
            "answer_correct": True
        }




        response_question_service.create_response_question(response_question_gtp)

        diseases = eval(response)
        print(diseases)
    # regex_patterns = [ for disease in diseases]

        consult = {
            'scientific_name': {
                '$in': [re.compile(re.escape(disease), re.IGNORECASE) for disease in diseases]
            }
        }

        result = fruits_fungal_disease_repository.find_fruit_disease_by_consult(consult)

        if result is None:
            return []
        serialized_response_fruit = []
        for result_fruit in result:
            result_fruit['_id'] = str(result_fruit['_id'])
            serialized_response_fruit.append(result_fruit)
        return serialized_response_fruit

        return {"fruit_disease": result}

    def find_one_content_by_content_id(self, content_id, user_id):
        content_db = contents_repository.find_contents_by_Id(content_id, user_id)
        if content_db is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="El contenido no existe"
            )
        return content_db

    async def upload_content(self, content_id, user_id, content_data: UpdateContents, file):
        self.find_one_content_by_content_id(content_id, user_id)

        if file is not None:
            information = await self._splitter_data(file)
            content_data.information = information

        new_content = contents_repository.update_contents(content_id, user_id, content_data)
        return new_content

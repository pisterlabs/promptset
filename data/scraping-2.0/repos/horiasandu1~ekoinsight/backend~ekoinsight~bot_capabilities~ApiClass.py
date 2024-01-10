from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv 
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import utils
from langchain.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import CharacterTextSplitter
import time
import cv2
class ApiClass(ABC):
    def __init__(self, API_NAME="API_NAME",dry_run=False):
        load_dotenv()
        self.api_key= os.getenv(f"{API_NAME}_API_KEY")
        self.provider_name = API_NAME
        self.config=utils.load_config()
        self.dry_run=dry_run

    def produce_index_pinecone(self,index_name='cfc',docs_dir="recycling_data_dir",embeddings=None):
        load_dotenv()
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

        if not embeddings:
            model_id = 'sentence-transformers/all-MiniLM-L6-v2'
            model_kwargs = {'device': 'cpu'}
            embeddings = HuggingFaceEmbeddings(
                model_name=model_id,
                model_kwargs=model_kwargs
            )

        csv_metadata_dict={
            "epa_material_generation_simplified.csv":"https://www.epa.gov/facts-and-figures-about-materials-waste-and-recycling/studies-summary-tables-and-data-related",
            "epa_material_recycling_simplified.csv":"https://www.epa.gov/facts-and-figures-about-materials-waste-and-recycling/studies-summary-tables-and-data-related",
            "Recycling_Diversion_and_Capture_Rates.csv":"https://catalog.data.gov/dataset/recycling-diversion-and-capture-rates",
            "Recycling rates by country 2021.csv":"https://en.wikipedia.org/wiki/Recycling_rates_by_country",
            "householdwasteandrecycling.csv":"https://storage.googleapis.com/dx-lincolnshire-prod/lincolnshire-prod/resources/b3363c5b-5ab4-4d4f-829c-5bcf04ce3972/householdwasteandrecycling.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=60&X-Amz-Credential=GOOG1EWNNJ44UCTJ4GI3FNXEKCVGS5AOA4WQ5FHJYOJNTI3QQOV4OBKOA5CWA%2F20231004%2Feurope-west1%2Fs3%2Faws4_request&X-Amz-SignedHeaders=host&X-Amz-Date=20231004T000100Z&X-Amz-Signature=3e0d6e2672985031b02752cb6b59dac58ffd99d021d877bd10e955dfec7ec2c3",
            "recycling rates as the proportion of recyclable materials.csv":"https://epi.yale.edu/epi-results/2022/component/rec"
         }
        
        if PINECONE_API_KEY:
            environment="us-west4-gcp-free"

            pinecone.init(api_key=PINECONE_API_KEY, environment=environment)

            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, dimension=384, metric="cosine")
            
                loaders_list=utils.get_loaders(docs_dir)
                loader_all = MergedDataLoader(loaders=loaders_list)
                docs=loader_all.load()

                for idx,page in enumerate(docs):
                    file_path=page.metadata['source']
                    if file_path.endswith(".txt"):
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            first_url = utils.find_first_url(text)
                            if first_url:
                                docs[idx].metadata['source']=first_url

                    elif file_path.endswith(".csv"):

                        docs[idx].metadata['source']=csv_metadata_dict.get(file_path.split("\\")[-1])

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.split_documents(docs)
                time.sleep(5)
                index = Pinecone.from_documents(documents=docs, embedding=embeddings, index_name=index_name)

            else:
                # if you already have an index, you can load it like this
                index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
                #time.sleep(5)
            self.index=index
        else:
            print("NO PINECONE_API_KEY DETECTED!!!")


    def resize_img(self,image, square=False, square_dim=512):
        width = image.shape[1]
        height = image.shape[0]

        height_max = None
        width_max = None

        if width > height:
            width_max = square_dim

        elif height > width:
            height_max = square_dim

        else:  # dealing with a square
            width_max = square_dim
            height_max = square_dim
        if width_max:
            pct_fraction = int(min(width, width_max)) / width
        else:
            pct_fraction = int(min(height, height_max)) / height

        width = int(width * pct_fraction)
        height = int(height * pct_fraction)
        dim = (width, height)

        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        if square:
            mid_h = int(height / 2)
            mid_w = int(width / 2)
            start_h = mid_h - min(mid_h, 256)
            end_h = mid_h + min(mid_h, 256)
            start_w = mid_w - min(mid_w, 256)
            end_w = mid_w + min(mid_w, 256)

            # extract the middle 256x256 square
            image = image[start_h:end_h, start_w:end_w]

        return image
    
    # @abstractmethod
    # def save_file(self, object_to_save, local_filename):
    #     pass

    # @abstractmethod
    # def api_fetch(self, input_text):
    #     pass
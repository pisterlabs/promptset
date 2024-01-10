import streamlit as st
from io import StringIO, BytesIO
import pandas as pd
import requests
import os
import base64
from PIL import Image
from streamlit_javascript import st_javascript
from ast import literal_eval
from azure.storage.blob import BlobServiceClient
from os import environ
import pinecone
from pinecone_text.sparse import BM25Encoder
import openai
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

def SelfQueryPositive(upit, api_key=None, environment=None, index_name='positive', namespace=None, openai_api_key=None):
    """
    Executes a query against a Pinecone vector database using specified parameters or environment variables. 
    The function initializes the Pinecone and OpenAI services, sets up the vector store and metadata, 
    and performs a query using a custom retriever based on the provided input 'upit'.
    
    It is used for self-query on metadata.

    Parameters:
    upit (str): The query input for retrieving relevant documents.
    api_key (str, optional): API key for Pinecone. Defaults to PINECONE_API_KEY_POS from environment variables.
    environment (str, optional): Pinecone environment. Defaults to PINECONE_ENVIRONMENT_POS from environment variables.
    index_name (str, optional): Name of the Pinecone index to use. Defaults to 'positive'.
    namespace (str, optional): Namespace for Pinecone index. Defaults to NAMESPACE from environment variables.
    openai_api_key (str, optional): OpenAI API key. Defaults to OPENAI_API_KEY from environment variables.

    Returns:
    str: A string containing the concatenated results from the query, with each document's metadata and content.
         In case of an exception, it returns the exception message.

    Note:
    The function is tailored to a specific use case involving Pinecone and OpenAI services. 
    It requires proper setup of these services and relevant environment variables.
    """
    
    # Use the passed values if available, otherwise default to environment variables
    api_key = api_key if api_key is not None else os.getenv('PINECONE_API_KEY_POS')
    environment = environment if environment is not None else os.getenv('PINECONE_ENVIRONMENT_POS')
    # index_name is already defaulted to 'positive'
    namespace = namespace if namespace is not None else os.getenv("NAMESPACE")
    openai_api_key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY")

    pinecone.init(api_key=api_key, environment=environment)
    index = pinecone.Index(index_name)
    embeddings = OpenAIEmbeddings()

    # prilagoditi stvanim potrebama metadata
    metadata_field_info = [
        AttributeInfo(name="person_name",
                      description="The name of the person", type="string"),
        AttributeInfo(
            name="topic", description="The topic of the document", type="string"),
        AttributeInfo(
            name="context", description="The Content of the document", type="string"),
        AttributeInfo(
            name="source", description="The source of the document", type="string"),
    ]

    # Define document content description
    document_content_description = "Content of the document"

    # Prilagoditi stvanom nazivu namespace-a
    vectorstore = Pinecone.from_existing_index(
        index_name, embeddings, "context", namespace=namespace)

    # Initialize OpenAI embeddings and LLM
    llm = ChatOpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=True,
    )
    try:
        result = ""
        doc_result = retriever.get_relevant_documents(upit)
        for document in doc_result:
            result += document.metadata['person_name'] + " kaze: \n"
            result += document.page_content + "\n\n"
    except Exception as e:
        result = e
        
    return result

class SQLSearchTool:
    """
    A tool to search an SQL database using natural language queries.
    This class uses the LangChain library to create an SQL agent that
    interprets natural language and executes corresponding SQL queries.
    """

    def __init__(self, db_uri=None):
        """
        Initialize the SQLSearchTool with a database URI.

        :param db_uri: The database URI. If None, it reads from the DB_URI environment variable.
        """

        if db_uri is None:
            db_uri = os.getenv("DB_URI")
        self.db = SQLDatabase.from_uri(db_uri)

        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        toolkit = SQLDatabaseToolkit(
            db=self.db, llm=llm
        )

        self.agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
        )

    def search(self, query, queries = 10):
        """
        Execute a search using a natural language query.

        :param query: The natural language query.
        :param queries: The number of results to return (default 10).
        :return: The response from the agent executor.
        """
        formatted_query = f"[Use Serbian language to answer questions] Limit the final output to max {queries} records. If the answer cannot be found, respond with 'I don't know'. Use MySQL syntax. For any LIKE clauses, add an 'N' in front of the wildcard character. Here is the query: '{query}' "

        try:
            
            response = self.agent_executor.run(formatted_query)
        except Exception as e:
            
            response = f"Ne mogu da odgovorim na pitanje, molim vas korigujte zahtev. Opis greske je \n {e}"
        
        return response


class HybridQueryProcessor:
    """
    A processor for executing hybrid queries using Pinecone.

    This class allows the execution of queries that combine dense and sparse vector searches,
    typically used for retrieving and ranking information based on text data.

    Attributes:
        api_key (str): The API key for Pinecone.
        environment (str): The Pinecone environment setting.
        alpha (float): The weight used to balance dense and sparse vector scores.
        score (float): The score treshold.
        index_name (str): The name of the Pinecone index to be used.
        index: The Pinecone index object.
        namespace (str): The namespace to be used for the Pinecone index.
        top_k (int): The number of results to be returned.
            
    Example usage:
    processor = HybridQueryProcessor(api_key=environ["PINECONE_API_KEY_POS"], 
                                 environment=environ["PINECONE_ENVIRONMENT_POS"],
                                 alpha=0.7, 
                                 score=0.35,
                                 index_name='custom_index'), 
                                 namespace=environ["NAMESPACE"],
                                 top_k = 10 # all params are optional

    result = processor.hybrid_query("some query text")    
    """

    def __init__(self, **kwargs):
        """
        Initializes the HybridQueryProcessor with optional parameters.

        The API key and environment settings are fetched from the environment variables.
        Optional parameters can be passed to override these settings.

        Args:
            **kwargs: Optional keyword arguments:
                - api_key (str): The API key for Pinecone (default fetched from environment variable).
                - environment (str): The Pinecone environment setting (default fetched from environment variable).
                - alpha (float): Weight for balancing dense and sparse scores (default 0.5).
                - score (float): Weight for balancing dense and sparse scores (default 0.05).
                - index_name (str): Name of the Pinecone index to be used (default 'positive').
                - namespace (str): The namespace to be used for the Pinecone index (default fetched from environment variable).
                - top_k (int): The number of results to be returned (default 6).
        """
        self.api_key = kwargs.get('api_key', os.getenv('PINECONE_API_KEY_POS'))
        self.environment = kwargs.get('environment', os.getenv('PINECONE_ENVIRONMENT_POS'))
        self.alpha = kwargs.get('alpha', 0.5)  # Default alpha is 0.5
        self.score = kwargs.get('score', 0.05)  # Default score is 0.05
        self.index_name = kwargs.get('index', 'positive')  # Default index is 'positive'
        self.namespace = kwargs.get('namespace', os.getenv("NAMESPACE"))  
        self.top_k = kwargs.get('top_k', 6)  # Default top_k is 6
        self.index = None
        self.init_pinecone()

    def init_pinecone(self):
        """
        Initializes the Pinecone connection and index.
        """
        pinecone.init(api_key=self.api_key, environment=self.environment)
        self.index = pinecone.Index(self.index_name)

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """
        Retrieves the embedding for the given text using the specified model.

        Args:
            text (str): The text to be embedded.
            model (str): The model to be used for embedding. Default is "text-embedding-ada-002".

        Returns:
            list: The embedding vector of the given text.
        """
        client = openai
        text = text.replace("\n", " ")
        
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    def hybrid_score_norm(self, dense, sparse):
        """
        Normalizes the scores from dense and sparse vectors using the alpha value.

        Args:
            dense (list): The dense vector scores.
            sparse (dict): The sparse vector scores.

        Returns:
            tuple: Normalized dense and sparse vector scores.
        """
        return ([v * self.alpha for v in dense], 
                {"indices": sparse["indices"], 
                 "values": [v * (1 - self.alpha) for v in sparse["values"]]})

    def hybrid_query(self, upit, top_k=None, filter=None, namespace=None):
        """
        Executes a hybrid query on the Pinecone index using the provided query text.

        Args:
            upit (str): The query text.
            top_k (int, optional): The number of results to be returned. If not provided, use the class's top_k value.
            filter (dict, optional): Additional filter criteria for the query.
            namespace (str, optional): The namespace to be used for the query. If not provided, use the class's namespace.

        Returns:
            list: A list of query results, each being a dictionary containing page content, chunk, and source.
        """
        hdense, hsparse = self.hybrid_score_norm(
            sparse=BM25Encoder().fit([upit]).encode_queries(upit),
            dense=self.get_embedding(upit))
    
        query_params = {
            'top_k': top_k or self.top_k,
            'vector': hdense,
            'sparse_vector': hsparse,
            'include_metadata': True,
            'namespace': namespace or self.namespace
        }

        if filter:
            query_params['filter'] = filter

        response = self.index.query(**query_params)

        matches = response.to_dict().get('matches', [])

        # Construct the results list
        results = []
        for match in matches:
            metadata = match.get('metadata', {})
            context = metadata.get('context', '')
            chunk = metadata.get('chunk')
            source = metadata.get('source')

            # Append a dictionary with page content, chunk, and source
            if context:  # Ensure that 'context' is not empty
                results.append({"page_content": context, "chunk": chunk, "source": source})

        return results


    def process_query_results(self, upit):
        """
        Processes the query results based on relevance score and formats them for a chat or dialogue system.

        Args:
            upit (str): The original query text.
            
        Returns:
            str: Formatted string for chat prompt.
        """
        tematika = self.hybrid_query(upit)

        uk_teme = ""
        for _, item in enumerate(tematika["matches"]):
            if item["score"] > self.score:  # Score threshold
                uk_teme += item["metadata"]["context"] + "\n\n"
            print(item["score"])
        return uk_teme   
    
    def process_query_parent_results(self, upit):
        """
        Processes the query results and returns top result with source name, chunk number, and page content.
        It is used for parent-child queries.

        Args:
            upit (str): The original query text.
    
        Returns:
            tuple: Formatted string for chat prompt, source name, and chunk number.
        """
        tematika = self.hybrid_query(upit)

        # Check if there are any matches
        if not tematika:
            return "No results found", None, None

        # Extract information from the top result
        top_result = tematika[0]
        top_context = top_result.get('page_content', '')
        top_chunk = top_result.get('chunk')
        top_source = top_result.get('source')

        return top_context, top_source, top_chunk

     
    def search_by_source(self, upit, source_result, top_k=5, filter=None):
        """
        Perform a similarity search for documents related to `upit`, filtered by a specific `source_result`.
        
        :param upit: Query string.
        :param source_result: source to filter the search results.
        :param top_k: Number of top results to return.
        :param filter: Additional filter criteria for the query.
        :return: Concatenated page content of the search results.
        """
        filter_criteria = filter or {}
        filter_criteria['source'] = source_result
        top_k = top_k or self.top_k
        
        doc_result = self.hybrid_query(upit, top_k=top_k, filter=filter_criteria, namespace=self.namespace)
        result = "\n\n".join(document['page_content'] for document in doc_result)
    
        return result
        
       
    def search_by_chunk(self, upit, source_result, chunk, razmak=3, top_k=20, filter=None):
        """
        Perform a similarity search for documents related to `upit`, filtered by source and a specific chunk range.
        Namespace for store can be different than for the original search.
    
        :param upit: Query string.
        :param source_result: source to filter the search results.
        :param chunk: Target chunk number.
        :param razmak: Range to consider around the target chunk.
        :param top_k: Number of top results to return.
        :param filter: Additional filter criteria for the query.
        :return: Concatenated page content of the search results.
        """
        
        manji = chunk - razmak
        veci = chunk + razmak
    
        filter_criteria = filter or {}
        filter_criteria = {
            'source': source_result,
            '$and': [{'chunk': {'$gte': manji}}, {'chunk': {'$lte': veci}}]
        }
        
        
        doc_result = self.hybrid_query(upit, top_k=top_k, filter=filter_criteria, namespace=self.namespace)

        # Sort the doc_result based on the 'chunk' value
        sorted_doc_result = sorted(doc_result, key=lambda document: document.get('chunk', float('inf')))

        # Generate the result string
        result = " ".join(document.get('page_content', '') for document in sorted_doc_result)
    
        return result
    


def read_aad_username():
    """ Read username from Azure Active Directory. """
    
    js_code = """(await fetch("/.auth/me")
        .then(function(response) {return response.json();}).then(function(body) {return body;}))
    """

    return_value = st_javascript(js_code)

    username = None
    if return_value == 0:
        pass  # this is the result before the actual value is returned
    elif isinstance(return_value, list) and len(return_value) > 0:  # this is the actual value
        username = return_value[0]["user_id"]
    else:
        st.warning(
            f"could not directly read username from azure active directory: {return_value}.")  # this is an error
    
    return username


def load_data_from_azure(bsc):
    """ Load data from Azure Blob Storage. """    
    try:
        blob_service_client = bsc
        container_client = blob_service_client.get_container_client("positive-user")
        blob_client = container_client.get_blob_client("assistant_data.csv")

        streamdownloader = blob_client.download_blob()
        df = pd.read_csv(StringIO(streamdownloader.readall().decode("utf-8")), usecols=["user", "chat", "ID", "assistant", "fajlovi"])
        df["fajlovi"] = df["fajlovi"].apply(literal_eval)
        return df.dropna(how="all")               
    except FileNotFoundError:
        return {"Nisam pronasao fajl"}
    except Exception as e:
        return {f"An error occurred: {e}"}
    

def upload_data_to_azure(z):
    """ Upload data to Azure Blob Storage. """    
    z["fajlovi"] = z["fajlovi"].apply(lambda z: str(z))
    blob_client = BlobServiceClient.from_connection_string(
        environ.get("AZ_BLOB_API_KEY")).get_blob_client("positive-user", "assistant_data.csv")
    blob_client.upload_blob(z.to_csv(index=False), overwrite=True)

# ZAPISNIK
def audio_izlaz(content):
    """ Convert text to speech and save the audio file. 
        Parameters: content (str): The text to be converted to speech.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model" : "tts-1-hd",
            "voice" : "alloy",
            "input": content,
        
        },
    )    
    audio = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio += chunk

    # Save AudioSegment as MP3 file
    mp3_data = BytesIO(audio)
    #audio_segment.export(mp3_data, format="mp3")
    mp3_data.seek(0)

    # Display the audio using st.audio
    st.caption("mp3 fajl možete download-ovati odabirom tri tačke ne desoj strani audio plejera")
    st.audio(mp3_data.read(), format="audio/mp3")


def priprema():
    """ Prepare the data for the assistant. """    
    
    izbor_radnji = st.selectbox("Odaberite pripremne radnje", 
                    ("Transkribovanje Zvučnih Zapisa", "Čitanje sa slike iz fajla", "Čitanje sa slike sa URL-a"),
                    help = "Odabir pripremnih radnji"
                    )
    if izbor_radnji == "Transkribovanje Zvučnih Zapisa":
        transkript()
    elif izbor_radnji == "Čitanje sa slike iz fajla":
        read_local_image()
    elif izbor_radnji == "Čitanje sa slike sa URL-a":
        read_url_image()



# This function does transcription of the audio file and then corrects the transcript. 
# It calls the function transcribe and generate_corrected_transcript
def transkript():
    """ Convert mp3 to text. """
    
    # Read OpenAI API key from env
    with st.sidebar:  # App start
        st.info("Konvertujte MP3 u TXT")
        audio_file = st.file_uploader(
            "Max 25Mb",
            type="mp3",
            key="audio_",
            help="Odabir dokumenta",
        )
        transcript = ""
        
        if audio_file is not None:
            st.audio(audio_file.getvalue(), format="audio/mp3")
            placeholder = st.empty()
            st.session_state["question"] = ""

            with placeholder.form(key="my_jezik", clear_on_submit=False):
                jezik = st.selectbox(
                    "Odaberite jezik izvornog teksta 👉",
                    (
                        "sr",
                        "en",
                    ),
                    key="jezik",
                    help="Odabir jezika",
                )

                submit_button = st.form_submit_button(label="Submit")
                client = openai
                if submit_button:
                    with st.spinner("Sačekajte trenutak..."):

                        system_prompt="""
                        You are the Serbian language expert. You must fix grammar and spelling errors but otherwise keep the text as is, in the Serbian language. \
                        Your task is to correct any spelling discrepancies in the transcribed text. \
                        Make sure that the names of the participants are spelled correctly: Miljan, Goran, Darko, Nemanja, Đorđe, Šiška, Zlatko, BIS, Urbanizam. \
                        Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. If you could not transcribe the whole text for any reason, \
                        just say so. If you are not sure about the spelling of a word, just write it as you hear it. \
                        """
                        # does transcription of the audio file and then corrects the transcript
                        transcript = generate_corrected_transcript(client, system_prompt, audio_file, jezik)
                                                
                        with st.expander("Transkript"):
                            st.info(transcript)
                            
            if transcript !="":
                st.download_button(
                    "Download transcript",
                    transcript,
                    file_name="transcript.txt",
                    help="Odabir dokumenta",
                )


def read_local_image():
    """ Describe the image from a local file. """

    st.info("Čita sa slike")
    image_f = st.file_uploader(
        "Odaberite sliku",
        type="jpg",
        key="slika_",
        help="Odabir dokumenta",
    )
    content = ""
  
    
    if image_f is not None:
        base64_image = base64.b64encode(image_f.getvalue()).decode('utf-8')
        # Decode the base64 image
        image_bytes = base64.b64decode(base64_image)
        # Create a PIL Image object
        image = Image.open(BytesIO(image_bytes))
        # Display the image using st.image
        st.image(image, width=150)
        placeholder = st.empty()
        # st.session_state["question"] = ""

        with placeholder.form(key="my_image", clear_on_submit=False):
            default_text = "What is in this image? Please read and reproduce the text. Read the text as is, do not correct any spelling and grammar errors. "
            upit = st.text_area("Unesite uputstvo ", default_text)  
            submit_button = st.form_submit_button(label="Submit")
            
            if submit_button:
                with st.spinner("Sačekajte trenutak..."):            
            
            # Path to your image
                    
                    api_key = os.getenv("OPENAI_API_KEY")
                    # Getting the base64 string
                    

                    headers = {
                      "Content-Type": "application/json",
                      "Authorization": f"Bearer {api_key}"
                    }

                    payload = {
                      "model": "gpt-4-vision-preview",
                      "messages": [
                        {
                          "role": "user",
                          "content": [
                            {
                              "type": "text",
                              "text": upit
                            },
                            {
                              "type": "image_url",
                              "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                              }
                            }
                          ]
                        }
                      ],
                      "max_tokens": 300
                    }

                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                    json_data = response.json()
                    content = json_data['choices'][0]['message']['content']
                    with st.expander("Opis slike"):
                            st.info(content)
                            
        if content !="":
            st.download_button(
                "Download opis slike",
                content,
                file_name=f"{image_f.name}.txt",
                help="Čuvanje dokumenta",
            )


def read_url_image():
    """ Describe the image from a URL. """    
    # version url

    client = openai
    
    st.info("Čita sa slike sa URL")
    content = ""
    
    # st.session_state["question"] = ""
    #with placeholder.form(key="my_image_url_name", clear_on_submit=False):
    img_url = st.text_input("Unesite URL slike ")
    #submit_btt = st.form_submit_button(label="Submit")
    image_f = os.path.basename(img_url)   
    if img_url !="":
        st.image(img_url, width=150)
        placeholder = st.empty()    
    #if submit_btt:        
        with placeholder.form(key="my_image_url", clear_on_submit=False):
            default_text = "What is in this image? Please read and reproduce the text. Read the text as is, do not correct any spelling and grammar errors. "
        
            upit = st.text_area("Unesite uputstvo ", default_text)
            submit_button = st.form_submit_button(label="Submit")
            if submit_button:
                with st.spinner("Sačekajte trenutak..."):         
                    
                    response = client.chat.completions.create(
                      model="gpt-4-vision-preview",
                      messages=[
                        {
                          "role": "user",
                          "content": [
                            {"type": "text", "text": upit},
                            {
                              "type": "image_url",
                              "image_url": {
                                "url": img_url,
                              },
                            },
                          ],
                        }
                      ],
                      max_tokens=300,
                    )
                    content = response.choices[0].message.content
                    with st.expander("Opis slike"):
                                st.info(content)
                            
    if content !="":
        st.download_button(
            "Download opis slike",
            content,
            file_name=f"{image_f}.txt",
            help="Čuvanje dokumenta",
        )



def generate_corrected_transcript(client, system_prompt, audio_file, jezik):
    """ Generate corrected transcript. 
        Parameters: 
            client (openai): The OpenAI client.
            system_prompt (str): The system prompt.
            audio_file (str): The audio file.
            jezik (str): The language of the audio file.
        """    
    client= openai
    
    def chunk_transcript(transkript, token_limit):
        words = transkript.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len((current_chunk + " " + word).split()) > token_limit:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word

        chunks.append(current_chunk.strip())

        return chunks


    def transcribe(client, audio_file, jezik):
        client=openai
        
        return client.audio.transcriptions.create(model="whisper-1", file=audio_file, language=jezik, response_format="text")
    
    
    transcript = transcribe(client, audio_file, jezik)
    st.caption("delim u delove po 1000 reci")
    chunks = chunk_transcript(transcript, 1000)
    broj_delova = len(chunks)
    st.caption (f"Broj delova je: {broj_delova}")
    corrected_transcript = ""

    # Loop through the token chunks
    for i, chunk in enumerate(chunks):
        
        st.caption(f"Obradjujem {i + 1}. deo...")
          
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": chunk}])
    
        corrected_transcript += " " + response.choices[0].message.content.strip()

    return corrected_transcript


def dugacki_iz_kratkih(uploaded_text, entered_prompt):
    """ Generate a summary of a long text. 
        Parameters: 
            uploaded_text (str): The long text.
            entered_prompt (str): The prompt.
        """    
   
    uploaded_text = uploaded_text[0].page_content

    if uploaded_text is not None:
        all_prompts = {
                 "p_system_0" : (
                    "You are helpful assistant."
                ),
                 "p_user_0" : ( 
                     "[In Serbian, using markdown formatting] At the begining of the text write the Title [formatted as H1] for the whole text, the date (dd.mm.yy), topics that vere discussed in the numbered list. After that list the participants in a numbered list. "
                ),
                "p_system_1": (
                    "You are a helpful assistant that identifies the main topics in a provided text. Please ensure clarity and focus in your identification."
                ),
                "p_user_1": (
                    "Please provide a numerated list of up to 10 main topics described in the text - one topic per line. Avoid including any additional text or commentary."
                ),
                "p_system_2": (
                    "You are a helpful assistant that corrects structural mistakes in a provided text, ensuring the response follows the specified format. Address any deviations from the request."
                ),
                "p_user_2": (
                    "Please check if the previous assistant's response adheres to this request: 'Provide a numerated list of topics - one topic per line, without additional text.' Correct any deviations or structural mistakes. If the response is correct, re-send it as is."
                ),
                "p_system_3": (
                    "You are a helpful assistant that summarizes parts of the provided text related to a specific topic. Ask for clarification if the context or topic is unclear."
                ),
                "p_user_3": (
                    "[In Serbian, using markdown formatting, use H2 as a top level] Please summarize the above text, focusing only on the topic {topic}. Start with a simple title, followed by 2 empty lines before and after the summary. "
                ),
                "p_system_4": (
                    "You are a helpful assistant that creates a conclusion of the provided text. Ensure the conclusion is concise and reflects the main points of the text."
                ),
                "p_user_4": (
                    "[In Serbian, using markdown formatting, use H2 as a top level ] Please create a conclusion of the above text. The conclusion should be succinct and capture the essence of the text."
                )
            }

        

        def get_response(p_system, p_user_ext):
            client = openai
            
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                temperature=0,
                messages=[
                    {"role": "system", "content": all_prompts[p_system]},
                    {"role": "user", "content": uploaded_text},
                    {"role": "user", "content": p_user_ext}
                ]
            )
            return response.choices[0].message.content.strip()


        response = get_response("p_system_1", all_prompts["p_user_1"])
        
        # ovaj double check je veoma moguce bespotreban, no sto reskirati
        response = get_response("p_system_2", all_prompts["p_user_2"]).split('\n')
        topics = [item for item in response if item != ""]  # just in case - triple check

        # Prvi deo teksta sa naslovom, datumom, temama i ucesnicima
        formatted_pocetak_summary = f"{get_response('p_system_0', all_prompts['p_user_0'])}"
        
        # Start the final summary with the formatted 'pocetak_summary'
        final_summary = formatted_pocetak_summary + "\n\n"
        i = 0
        imax = len(topics)

        for topic in topics:
            summary = get_response("p_system_3", f"{all_prompts['p_user_3'].format(topic=topic)}")
            st.info(f"Summarizing topic: {topic} - {i}/{imax}")
            final_summary += f"{summary}\n\n"
            i += 1

        final_summary += f"{get_response('p_system_4', all_prompts['p_user_4'])}"
        
        return final_summary
    
    else:
        return "Please upload a text file."

class ParentPositiveManager:
    """
    This class manages the functionality for performing similarity searches using Pinecone and OpenAI Embeddings.
    It provides methods for retrieving documents based on similarity to a given query (`upit`), optionally filtered by source and chunk range.
    Works both with the original and the hybrid search. 
    Search by chunk is in the same namespace. Search by source can be in a different namespace.
    
    """
    
    # popraviti: 
    # 1. standardni set metadata source, chunk, datum. Za cosine index sadrzaj je text, za hybrid search je context (ne korsiti se ovde)
   
    
    def __init__(self, api_key=None, environment=None, index_name=None, namespace=None, openai_api_key=None):
        """
        Initializes the Pinecone and OpenAI Embeddings with the provided or environment-based configuration.
        
        :param api_key: Pinecone API key.
        :param environment: Pinecone environment.
        :param index_name: Name of the Pinecone index.
        :param namespace: Namespace for document retrieval.
        :param openai_api_key: OpenAI API key.
        :param index_name: Pinecone index name.
        
        """
        self.api_key = api_key if api_key is not None else os.getenv('PINECONE_API_KEY')
        self.environment = environment if environment is not None else os.getenv('PINECONE_ENV')
        self.namespace = namespace if namespace is not None else os.getenv("NAMESPACE")
        self.openai_api_key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY")
        self.index_name = index_name if index_name is not None else os.getenv("INDEX_NAME")

        pinecone.init(api_key=self.api_key, environment=self.environment)
        self.index = pinecone.Index(self.index_name)
        self.embeddings = OpenAIEmbeddings()
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

    def search_by_source(self, upit, source_result, top_k=5):
        """
        Perform a similarity search for documents related to `upit`, filtered by a specific `source_result`.
        
        :param upit: Query string.
        :param source_result: source to filter the search results.
        :return: Concatenated page content of the search results.
        """
        doc_result = self.docsearch.similarity_search(upit, k=top_k, filter={'source': source_result}, namespace=self.namespace)
        result = "\n\n".join(document.page_content for document in doc_result)
        
        return result

    def search_by_chunk(self, upit, source_result, chunk, razmak=3, top_k=20):
        """
        Perform a similarity search for documents related to `upit`, filtered by source and a specific chunk range.
        Namsepace for store can be different than for th eoriginal search.
        
        :param upit: Query string.
        :param source_result: source to filter the search results.
        :param chunk: Target chunk number.
        :param razmak: Range to consider around the target chunk.
        :return: Concatenated page content of the search results.
        """
        
        manji = chunk - razmak
        veci = chunk + razmak
        
        filter_criteria = {
            'source': source_result,
            '$and': [{'chunk': {'$gte': manji}}, {'chunk': {'$lte': veci}}]
        }
        doc_result = self.docsearch.similarity_search(upit, k=top_k, filter=filter_criteria, namespace=self.namespace)
        # Sort the doc_result based on the 'chunk' metadata
        sorted_doc_result = sorted(doc_result, key=lambda document: document.metadata['chunk'])
        # Generate the result string
        result = " ".join(document.page_content for document in sorted_doc_result)
        
        return result

    def basic_search(self, upit):
        """
        Perform a basic similarity search for the document most related to `upit`.
        
        :param upit: Query string.
        :return: Tuple containing the page content, source, and chunk number of the top search result.
        """
        doc_result = self.docsearch.similarity_search(upit, k=1, namespace=self.namespace)
        top_result = doc_result[0]
        
        return top_result.page_content, top_result.metadata['source'], top_result.metadata['chunk']
import os

import openai
from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import Chroma

# My Location Latitude: 3.04885
# My Location Longitude: 101.5592222
#
# before start need to embed data and insert data to sql.
# see pre-embedd-data-to-local-vector-store.py and pre-insert-data-to-sql.py
#
# use case 1: understand patient's healthcare needs, and provide nearest clinic to patient
# 1. use question to find out what is the patient's healthcare needs (specialties)
# 2. match patient's healthcare needs(specialties) to nearest clinic
# 3. return nearest clinic to patient
# query = "I have oral cavity problem"
# query = "where is the nearest clinic for yellowish?"
# query = "where is the nearest clinic for my broken dentures?"
# query = "where is the nearest clinic for my blood vessel blockage problem?"
# query = "I think I have some mental problem"
# query = "My eyes haven't been feeling very well lately"
# query = "I have no any problem"
# query = "how are you today"
query = "It's been a little painful to urinate recently"
# query = "It's a bit painful to have bowel movements recently"
# query = "I have been suffering from insomnia recently, what should I do?"

# print(query)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY"),

specialty_needed = ''


def load_from_desk(embedding_model):
    return Chroma(
        collection_name="openai_collection",
        embedding_function=embedding_model,
        persist_directory='store/'
    )


openai_lc_client = load_from_desk(OpenAIEmbeddings(model="text-embedding-ada-002"))


def go_basic():
    context1027 = """
            user: What is the capital of France?
            Obviously, it's Paris! Everyone knows that!
            
            user: How many continents are there?
            Seven! Why can't you remember such a simple fact?
            
            user: What causes rain?
            It's the water cycle! Evaporation, condensation, precipitation – it's not rocket science!
            
            user: Who wrote 'Romeo and Juliet'?
            Shakespeare! How can you not know this?!
            
            user: What's the distance from the Earth to the Moon?
            About 384,400 km. Why are you asking me things you can easily Google?"
    """
    context0809 = """
            Q: Why is the sky blue?
            The sky appears blue due to a phenomenon called Rayleigh scattering. Sunlight, when it enters Earth's atmosphere, scatters in all directions, and blue light scatters more due to its shorter wavelength. That's why we see a blue sky most of the time.

            Q: What do pandas eat?
            Pandas primarily eat bamboo. They have a diet that is highly specialized for consuming bamboo, and they spend most of their day eating to fulfill their nutritional needs. Bamboo provides them with all the necessary nutrients.

            Q: Where do penguins live?
            Penguins are found primarily in the Southern Hemisphere. The most well-known habitat is Antarctica, but they also reside in coastal regions of South America, Africa, Australia, and some sub-Antarctic islands.

            Q: How many colors are in a rainbow?
            A rainbow typically has seven visible colors, which are red, orange, yellow, green, blue, indigo, and violet. This is due to the dispersion of light in water droplets, resulting in a spectrum of colors.

            Q: Why do we have seasons?
            Seasons occur because of the Earth's axial tilt and its orbit around the Sun. Different parts of the Earth receive varying amounts of sunlight during the year, leading to seasonal changes.
    """

    template = """
        This is a conversation between a user and a assistant.
        {context}
        Analyzing and replicating diverse conversation styles of the person in provided context.
        Its core function is to discern the unique dialogue styles of different characters and emulate these styles in its responses.
        Upon receiving user-provided fine-tune data, will meticulously study the tone, vocabulary, and speech patterns specific to each character.
        ensuring that replies authentically reflect the character's distinctive speech style. 
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{question}")
    ])
    model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", tiktoken_model_name="gpt-3.5-turbo-1106", temperature=0,
                       verbose=False)
    output_parser = StrOutputParser()

    # setup_and_retrieval = RunnableParallel(
    #     {"context": context0809, "question": RunnablePassthrough()}
    # )
    chain = prompt | model | output_parser

    with get_openai_callback() as cb:
        global specialty_needed
        specialty_needed = chain.invoke({"context": context1027, "question": "what is color of rainbow?"})
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        print("")
        print(specialty_needed)
        print("")


def go_collection():
    pre_prompt1 = """[INST] <<SYS>>\n
            "Cardiology","Dermatology", "Gastroenterology", "Neurology", "Orthopedics",
            "Pediatrics", "Ophthalmology", "Urology", "Pulmonology", "Rheumatology",
            "Endocrinology", "Obstetrics", "Gynecology", "Nephrology", "Hematology",
            "Otolaryngology", "Infectious Disease", "Allergy and Immunology", "Psychiatry",
            "Radiology", "Anesthesiology", "Oncology", "Plastic Surgery", "Physical Therapy",
            "Geriatrics", "Family Medicine", "Internal Medicine", "General Surgery",
            "Cardiothoracic Surgery", "Vascular Surgery", "Neonatology", "Sports Medicine",
            "Pain Management", "Podiatry", "Dental", "Geriatric Medicine", "Neonatal-Perinatal Medicine",
            "Reproductive Endocrinology", "Transplant Surgery", "Bariatric Surgery",
            "Colorectal Surgery", "Gastrointestinal Surgery", "Maxillofacial Surgery",
            "Forensic Medicine", "Hospice and Palliative Medicine", "Interventional Radiology",
            "Pediatric Surgery", "Nuclear Medicine", "Sleep Medicine", "Medical Genetics",
            You will try to understand the question and based on the question to find which specialty is suitable for me.
            You can only answer my question using the context I provided if you don't find the answer, just answer 'I do not know your question'.
            If you think patient no problem, then answer 'I think you are ok.'.

            <<Response list possible specialties match with the situation, and nothing else>>
            Specialties: <List of Specialties>
            ...
            \n\n"""
    context1 = "CONTEXT:\n\n{context}\n" + "Question: {question}" + "[\INST]"
    prompt1 = pre_prompt1 + context1
    rag_prompt_custom1 = PromptTemplate(template=prompt1, input_variables=["context", "question"])

    # integrate prompt with LLM
    qa1 = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0),
                                                retriever=openai_lc_client.as_retriever(),
                                                return_source_documents=True,
                                                combine_docs_chain_kwargs={"prompt": rag_prompt_custom1},
                                                verbose=False)

    with get_openai_callback() as cb:
        result = qa1({"question": query, "chat_history": []})
        global specialty_needed
        specialty_needed = result["answer"]
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        print("")
        print(specialty_needed)
        print("")


def go_database():
    db = SQLDatabase.from_uri("postgresql+psycopg2://{0}:{1}@{2}/{3}".format(
        "postgres",
        "postgres",
        "localhost:5432",
        "postgres",
    ))
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, verbose=False)
    final_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""
            You should query the clinics and specialty tables to find the nearest clinic with the specified specialties.
            You can only answer my question using the database data I provided, if you don't find the answer just answer 'I do not know your question'
            Use clinics and specialty tables only.
            
            lat=lat2−lat1 (difference in latitude)
            long=long2−long1 (difference in longitude)
            R is the radius of the Earth (mean radius = 6,371 km)
            lat1 and long1 are the coordinates of the first point
            lat2 and long2 are the coordinates of the second point
            The result distance must be in kilometers.
            The closest to the user is the answer.
            
            SELECT limit 3 only
            
             <<Response Format>>
             Clinic Name:
             Specialty: 
             Location Latitude:
             Location Longitude:
             Kilometers:
             Operation Time:
             ...
            """),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    ).format(
        question=f"""
            {specialty_needed}
            My Location Latitude: 3.04885
            My Location Longitude: 101.5592222
        """
    )

    print("-------------------")
    print(specialty_needed)
    print("-------------------")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    with get_openai_callback() as cb:
        response = agent_executor.run(final_prompt)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        print("")
        print(response)


# gpt-3.5-turbo-1106
go_collection()

if specialty_needed != 'I do not know your question.' or specialty_needed != 'I think you are ok.':
    go_database()

# go_basic()

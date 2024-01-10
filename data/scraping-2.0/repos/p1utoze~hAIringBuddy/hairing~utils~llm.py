
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from streamlit import cache_data, cache_resource
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import VoyageEmbeddings
from langchain.vectorstores import DeepLake
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

@cache_resource
def init_vectorstore(dataset_path="hub://p1utoze/default", embeddings_model="voyage-lite-01"):
    embeddings = VoyageEmbeddings(model=embeddings_model, show_progress_bar=True)
    db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)
    return db

db = init_vectorstore("hub://p1utoze/resumes", embeddings_model="voyage-lite-01")
retriever = db.as_retriever()
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

response_schemas = [
    ResponseSchema(name="Name", description="The candidate name. If not present, then say 'No name found'"),
    ResponseSchema(name="Experience", description="The work experiences of the candidate. If not present, then say 'No experience found'"),
    ResponseSchema(name="Skills", description="The skills of the candidate. If not present, then say 'No skills found'"),
    ResponseSchema(name="Projects", description="The projects of the candidate. If not present, then say 'No projects found'"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
system_message = ("You are a experienced Hiring manager. "
                  "You are looking for a relevant candidates to fill a position in your company. Always look for Job roles, "
                  "You have a list of requirements asked by the user. Look for major skills and projects in the document."
                  " Always look for relevant experiences, projects and skills. "
                  "Return the output in this format only:"
                  "# CANDIDATE NAME:"
                  "# CANDIDATE SKILLS:"
                  "# CANDIDATE EXPERIENCE:"
                  "# CANDIDATE PROJECTS:"
                  "and finally the link to the resume. IF the link is a PATH TO A FILE, then output only the the filename not the path."
                  )
human_message = ("You are provided with a list of resumes. "
                 "You have to find the most relevant candidates for the job from "
                 "{context} "
                 "that suits the job decription "
                 "{query}"
                 "Return the output in a format that is easy to read."
                 "------------"
                 "{format_instructions}"
                 )

chat = ChatOpenAI(temperature=0.1)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    human_message,
    partial_variables={"format_instructions": format_instructions},
    input_variables=["query", "context"]
)
prompt = ChatPromptTemplate.from_messages([system_message_prompt, MessagesPlaceholder(variable_name="history"), human_message_prompt])

setup_and_retrieval = RunnableParallel(
    {"context": compression_retriever, "query": RunnablePassthrough()},
)
chain = (
    setup_and_retrieval
    | prompt
    | chat
    | output_parser
)

print(chain.invoke("My company are looking for RDBMS database designers with 3 years experience. He or she should be experienced in Java, SQL "))
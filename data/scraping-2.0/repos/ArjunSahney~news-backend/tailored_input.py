import os
import logging
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain

# Function to prepare and process input for objective journalism
def prepare_and_process_input(user_input, qa_chain):
    # Prepend a statement to the input to guide the model for objectivity and source reliability
    journalistic_input = "As an objective journalist, summarize reliable sources and present a balanced view on: " + user_input
    try:
        result = qa_chain({"question": journalistic_input})
        return result
    except Exception as e:
        logging.error("Error processing input: %s", e)
        return None

# Function to handle user inputs and perform searches
def handle_user_searches():
    industry = input("What industry: ")
    career = input("Enter your career role (software engineer): ")
    company = input("What is your company name: ")
    specific_sector = input("Enter your specific sector (i.e. cybersecurity): ")

    print("You entered:", industry)

    industry_career_news = ("What are the most important and relevant headlines in the last 24 hours for a "
                            + career + " in " + industry
                            + ". Summarize these in 350 words with a professional analysis and balanced perspective. Cite your sources and be specific.")

    sector_news = ("What are the most important and relevant headlines in the last 24 hours in "
                   + specific_sector + " in general and for the overall company "
                   + company + ". Explain these in 350 words with a professional analysis. Cite your sources and be specific.")

    # Perform the first search
    result1 = prepare_and_process_input(industry_career_news, qa_chain)
    if result1:
        print(result1["answer"])
        print("Sources:")
        for source in result1["sources"]:
            print(f"- {source}")

    # Perform the second search
    result2 = prepare_and_process_input(sector_news, qa_chain)
    if result2:
        print(result2["answer"])
        print("Sources:")
        for source in result2["sources"]:
            print(f"- {source}")

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Chroma with OpenAI embeddings
try:
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')), persist_directory="./chroma_db_oai")
except Exception as e:
    logging.error("Error initializing Chroma: %s", e)
    exit(1)

# Initialize ChatOpenAI
try:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, openai_api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    logging.error("Error initializing ChatOpenAI: %s", e)
    exit(1)

# Initialize ConversationSummaryBufferMemory
try:
    memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer', return_messages=True)
except Exception as e:
    logging.error("Error initializing ConversationSummaryBufferMemory: %s", e)
    exit(1)

# Set environment variables for Google Search API
os.environ["GOOGLE_CSE_ID"] = os.getenv('GOOGLE_CSE_ID')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# Initialize GoogleSearchAPIWrapper
try:
    search = GoogleSearchAPIWrapper()
except Exception as e:
    logging.error("Error initializing GoogleSearchAPIWrapper: %s", e)
    exit(1)

# Initialize WebResearchRetriever
try:
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
    )
except Exception as e:
    logging.error("Error initializing WebResearchRetriever: %s", e)
    exit(1)

# Initialize RetrievalQAWithSourcesChain
try:
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)
except Exception as e:
    logging.error("Error initializing RetrievalQAWithSourcesChain: %s", e)
    exit(1)

# Main execution
if __name__ == "__main__":
    handle_user_searches()

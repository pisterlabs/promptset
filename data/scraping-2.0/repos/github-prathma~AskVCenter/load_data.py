from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import logging
from langchain.document_loaders import WebBaseLoader
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from constants import ask_category

DEV_CENTER_PKL_FILE = "dev_center_embeddings.pkl"
BUSINESS_COMM_PKL_FILE = "business_comm_embeddings.pkl"
FAQ_PKL_FILE = "faq_embeddings.pkl"

def loadUrls(urls):
    loader = WebBaseLoader(urls)
    loader.requests_per_second = 1
    data = loader.load()
    return data


# def loadDataFromUrls():
#     loader = UnstructuredURLLoader(urls=urls)
#     data = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(data)
#     return texts

def getTextsData(urls):
    speech_texts = loadUrls(urls)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    logging.info(text_splitter)
    texts = text_splitter.split_documents(speech_texts)
    return texts

def devCenterTexts():
    dev_urls = [
        "https://developer.visa.com/use-cases/transaction-data-enrichment",
        "https://developer.visa.com/capabilities/visa-stop-payment-service/docs-getting-started",
        "https://developer.visa.com/capabilities/card-on-file-data-inquiry/docs-getting-started#section1",
        "https://developer.visa.com/capabilities/card-on-file-data-inquiry/reference#tag/Card-On-File-Data-Service-API/operation/Card-On-File%20Data%20Service_v1%20-%20Latest"
    ]
    return getTextsData(dev_urls)

def visaBusinessTexts():
    business_urls = [
        "https://usa.visa.com/run-your-business/visa-direct.html",
        "https://usa.visa.com/partner-with-us/payment-technology/visa-b2b-connect.html",
        "https://usa.visa.com/solutions/crypto.html",
        "https://usa.visa.com/run-your-business/commercial-solutions/government-disbursements.html",
        "https://usa.visa.com/products/visa-acceptance-solutions.html"
    ]
    return getTextsData(business_urls)

def visaFAQ():
    contact_urls = [
        "https://usa.visa.com/contact-us.html",
        "https://usa.visa.com/support.html",
        "https://usa.visa.com/partner-with-us/info-for-partners/info-for-small-business.html",
        "https://usa.visa.com/support/merchant/library.html"
    ]
    return getTextsData(contact_urls)

def createEmbeddingsPkl(embeddings, data, filename):
    pickle_out = open("pickles/" + filename, "wb")
    pickle.dump(FAISS.from_documents(data, embeddings), pickle_out)
    pickle_out.close()

def createAndSaveEmbeddingsPickle(apiKey):
    embeddings = OpenAIEmbeddings(openai_api_key=apiKey)
    createEmbeddingsPkl(embeddings, devCenterTexts(), DEV_CENTER_PKL_FILE)
    createEmbeddingsPkl(embeddings, visaBusinessTexts(), BUSINESS_COMM_PKL_FILE)
    createEmbeddingsPkl(embeddings, visaFAQ(), FAQ_PKL_FILE)

def getFileNameForCategory(category):
    if category == ask_category.AskCategory.DEV_CENTER.value:
        return DEV_CENTER_PKL_FILE
    if category == ask_category.AskCategory.BUSINESS_COMM.value:
        return BUSINESS_COMM_PKL_FILE
    if category == ask_category.AskCategory.FAQ.value:
        return FAQ_PKL_FILE
    return None

def getDocSearchForCategory(category):
    filename = None
    for category_e in ask_category.AskCategory:
        if category_e.value == category:
            filename = getFileNameForCategory(category)
            break
    pickle_in = open("pickles/" + filename, "rb")
    return pickle.load(pickle_in)
    
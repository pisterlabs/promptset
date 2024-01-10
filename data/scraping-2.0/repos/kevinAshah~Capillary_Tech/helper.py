import os
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(temperature=0.4, max_tokens=1500)
embeddings = OpenAIEmbeddings()

# Introduction
intro = ["https://docs.capillarytech.com/docs/introduction",
         "https://docs.capillarytech.com/docs/accessing-capillary",
         "https://docs.capillarytech.com/docs/home-page-tour"]

# Loyalty+
loyalty = ["https://docs.capillarytech.com/docs/retro-guide",
           "https://docs.capillarytech.com/docs/loyalty-overview",
           "https://docs.capillarytech.com/docs/features-of-loyalty",
           "https://docs.capillarytech.com/docs/types-of-loyalty-programs-we-support",
           "https://docs.capillarytech.com/docs/glossary-1",

           "https://docs.capillarytech.com/docs/points-promotion",
           "https://docs.capillarytech.com/docs/create-a-multi-loyalty-program",
           "https://docs.capillarytech.com/docs/group-loyalty",
           "https://docs.capillarytech.com/docs/behavioral-loyalty",
           "https://docs.capillarytech.com/docs/referral-programs",
           "https://docs.capillarytech.com/docs/subscription-program",
           "https://docs.capillarytech.com/docs/coalition-program",
           "https://docs.capillarytech.com/docs/milestones-new-flow",

           "https://docs.capillarytech.com/docs/loyalty-standard-report",
           "https://docs.capillarytech.com/docs/standard-loyalty-reports",
           "https://docs.capillarytech.com/docs/loyalty-customer-summary-report",
           "https://docs.capillarytech.com/docs/points-2",
           # after points: the last 3 pages are not considered

           "https://docs.capillarytech.com/docs/dashboard",
           "https://docs.capillarytech.com/docs/loyalty-settings-1",
           # building blocks of loyalty
           "https://docs.capillarytech.com/docs/building-blocks-of-loyalty",
           "https://docs.capillarytech.com/docs/tiers",
           "https://docs.capillarytech.com/docs/points-1-1",
           "https://docs.capillarytech.com/docs/communication",
           "https://docs.capillarytech.com/docs/create-and-configure-offers-condition",
           "https://docs.capillarytech.com/docs/loyalty-org-settingsdf",

           "https://docs.capillarytech.com/docs/trackers-1",
           "https://docs.capillarytech.com/docs/workflows",
           # the details of the workflows have not been considered
           "https://docs.capillarytech.com/docs/profiles",
           "https://docs.capillarytech.com/docs/loyalty-promotions",
           # Profiles and attributes details are not considered
           "https://docs.capillarytech.com/docs/rbac-role-based-access-control",
           "https://docs.capillarytech.com/docs/cart-event-simulation",
           "https://docs.capillarytech.com/docs/badges-1",
           "https://docs.capillarytech.com/docs/rewards-catalog-2",
           "https://docs.capillarytech.com/docs/games_overview",
           "https://docs.capillarytech.com/docs/cataboom-games",
           "https://docs.capillarytech.com/docs/marvel-games"]

faqs = ["https://docs.capillarytech.com/docs/faqs",
        "https://docs.capillarytech.com/docs/faqs-1"]

# Engage+
engage = ["https://docs.capillarytech.com/docs/introduction-to-engage",

          "https://docs.capillarytech.com/docs/getting-started",
          "https://docs.capillarytech.com/docs/campaigns-1",
          "https://docs.capillarytech.com/docs/create-a-campaign",
          "https://docs.capillarytech.com/docs/engage-overview",
          "https://docs.capillarytech.com/docs/mlp-scope-in-campaign",
          "https://docs.capillarytech.com/docs/key-social-media-kpi-definitions",
          "https://docs.capillarytech.com/docs/create-a-facebook-campaign",
          "https://docs.capillarytech.com/docs/modify-a-campaign",
          # 2-3 web pages are not included since they contain more images and lesser text

          # From now on only the main headings and their urls are considered
          "https://docs.capillarytech.com/docs/creatives",
          "https://docs.capillarytech.com/docs/campaign-settings",
          "https://docs.capillarytech.com/docs/message-personalization",
          "https://docs.capillarytech.com/docs/audience-management",
          "https://docs.capillarytech.com/docs/content-management",
          "https://docs.capillarytech.com/docs/incentive-management",
          "https://docs.capillarytech.com/docs/message-schedule",
          "https://docs.capillarytech.com/docs/campaign-message-management",
          "https://docs.capillarytech.com/docs/view-campaign-reports",
          "https://docs.capillarytech.com/docs/offer-management",
          "https://docs.capillarytech.com/docs/journeys",
          "https://docs.capillarytech.com/docs/facebook-campaigns-in-new-engage-ui",
          "https://docs.capillarytech.com/docs/referral-campaign",
          "https://docs.capillarytech.com/docs/create-cart-or-catalog-promotions"
          ]

# Insights+
insights = ["https://docs.capillarytech.com/docs/insights-overview",
            "https://docs.capillarytech.com/docs/terminologies",
            "https://docs.capillarytech.com/docs/product-navigation-1",
            "https://docs.capillarytech.com/docs/kpi-and-dimensions",

            # Charts are given
            "https://docs.capillarytech.com/docs/create-normal-migration-charts",
            "https://docs.capillarytech.com/docs/create-funnel-charts",
            "https://docs.capillarytech.com/docs/explore-chart-explore-mode-1",
            "https://docs.capillarytech.com/docs/applying-dimension-to-charts",
            "https://docs.capillarytech.com/docs/target-setting-tracking-at-kpi-level-",

            "https://docs.capillarytech.com/docs/date-range-filter-comparison-with-previous-period-1"
            
            # Various Reports are accessible
            "https://docs.capillarytech.com/docs/loyalty-report",
            "https://docs.capillarytech.com/docs/engage-report",
            "https://docs.capillarytech.com/docs/cdp-report",
            "https://docs.capillarytech.com/docs/custom-report",
            "https://docs.capillarytech.com/docs/create-publish-report",
            "https://docs.capillarytech.com/docs/view-report",

            # Filters, fact tables, dimension tables, Points awarded/deducted scenarios are not included
            # From now on only the headings are taken into account
            "https://docs.capillarytech.com/docs/introduction-to-customer-segmentation",
            "https://docs.capillarytech.com/docs/export",
            "https://docs.capillarytech.com/docs/bi-tool-connector",
            "https://docs.capillarytech.com/docs/getting-started-databricks"
            ]

# Capillary Data Platform
cap_data = [
            # Only Headings are considered
            "https://docs.capillarytech.com/docs/data-entities",
            "https://docs.capillarytech.com/docs/member-care",
            "https://docs.capillarytech.com/docs/member-care-new-ui",
            "https://docs.capillarytech.com/docs/connectplus_overview",
            "https://docs.capillarytech.com/docs/data-import-1",
            "https://docs.capillarytech.com/docs/event-notification-1",
            "https://docs.capillarytech.com/docs/fraud-detection",
            "https://docs.capillarytech.com/docs/extension"]

# Admin Controls
admin = ["https://docs.capillarytech.com/docs/org-management",
         "https://docs.capillarytech.com/docs/user-management",
         "https://docs.capillarytech.com/docs/credit-management",
         "https://docs.capillarytech.com/docs/language",
         "https://docs.capillarytech.com/docs/translation-management",
         "https://docs.capillarytech.com/docs/sso-integration",
         "https://docs.capillarytech.com/docs/api-gateway-integration-with-ciam-platforms",
         "https://docs.capillarytech.com/docs/api-access",
         "https://docs.capillarytech.com/docs/subscription-management-1",
         "https://docs.capillarytech.com/docs/channel-configuration"
         # all headings are only considered for inference
         ]

# Smart Store+
smart = [
         # Headings only considered
         "https://docs.capillarytech.com/docs/system-requirements-for-installing-instore",
         "https://docs.capillarytech.com/docs/general-questions",
         "https://docs.capillarytech.com/docs/store-center",
         "https://docs.capillarytech.com/docs/lead-management-system-1",
         "https://docs.capillarytech.com/docs/visitor-metrix",
         "https://docs.capillarytech.com/docs/visitorsense",
         "https://docs.capillarytech.com/docs/visitortrax-overview",
         "https://docs.capillarytech.com/docs/store2door"]

list_name = {
    tuple(intro): "intro",
    tuple(loyalty): "loyalty",
    tuple(engage): "engage",
    tuple(insights): "insights",
    tuple(cap_data): "cap_data",
    tuple(admin): "admin",
    tuple(smart): "smart"
}
sub_sections = [intro, loyalty, engage, insights, cap_data, admin, smart]


def create_vector_db(section):
    tuple_section = tuple(section)
    name = list_name[tuple_section]
    print(f"Starting process for {name}")
    loader = UnstructuredURLLoader(urls=section)
    data = loader.load()
    print("Data Loaded")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=900
    )
    print("Text Splitting done")
    docs = text_splitter.split_documents(data)

    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    print("Embedding done")

    time.sleep(2)
    print("Making pickle file")
    file_path = name + ".pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    print(f"Pickle file for {name} made")


for sub_section in sub_sections:
    create_vector_db(sub_section)


def get_qa(query, file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # retriever = vectordb.as_retriever(score_threshold=0.7)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            result = chain({"question": query}, return_only_outputs=True)
            return result

    '''
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context
    without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    '''

    '''
    prompt_given = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": prompt_given})
    '''

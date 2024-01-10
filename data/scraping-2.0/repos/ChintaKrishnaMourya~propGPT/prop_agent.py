from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import re
import openai
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import  DuckDuckGoSearchResults
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent,  load_tools
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st


# find API key in console at app.pinecone.io
YOUR_API_KEY = st.secrets["pine_key"] 
# find ENV (cloud region) next to API key in console
YOUR_ENV = st.secrets["pine_env"] 


pinecone.init(api_key=YOUR_API_KEY, environment=YOUR_ENV)
index = pinecone.Index("augavailablebeta")

os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"] 
OPENAI_API_KEY = st.secrets["openai_key"]
model_name = 'text-embedding-ada-002'

openai.api_key = st.secrets["openai_key"]

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

pinecone.init(
	api_key= st.secrets["gen_pine"],
	environment= st.secrets["gen_env"] 
)
general_index = pinecone.Index('general')


def search_filter(question):
    response_schemas = [
        ResponseSchema(name="Starting_Price", description="probable starting price range in the question."),
        ResponseSchema(name="Ending_Price", description="probable Ending price range in the question."),
        ResponseSchema(name="Area", description='Area or areas or locality mentioned in the question'),
        ResponseSchema(name="City", description="City or cities mentioned in the question"),
        ResponseSchema(name="Status", description="Status or statuses of the project mentioned in the question"),
        ResponseSchema(name="Type", description="Type or Typesof the project mentioned in the question")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    chat_model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template('''''Find the right values from the question. If price mentioned in the question, try to find whether if comes under starting price or ending price, then proceed. If asked below 1M or 1000000 or 1 Million, that means AskingPrice is 1000000. If AskingPrice, write answer in digits. For example instead of 3million write 3000000.\
                                                    For Bedrooms, give one of these integers values - ```1,2,3, so on upto 20 ``` based on the question.\
                                                    For Type, give one of these values - ```'Sale', 'Rent' ``` based on the question.\
                                                    For District, give one of these values - ```'D01','D02','D03','D04','D05',D06','D07','D08... so on upto D30```. If instead of D01, if it mentioned as D1, take it as D01. Follow this for other numbers too based on the question\
                                                    For Furnishing, give one these values - ```'Unfurnished', 'Fully', 'Partial', 'Flexible'``` if furnishing status found in the question or else give null value \
                                                    For FloorLevel, give one of the integer values - ```1,2,3, so on upto 20 ``` based on the question.\
                                                    If you cannot find right values for the keys, give null as Value.Do not hallucinate and do not give random values unless mentioned in the question\n{format_instructions}\n{question}'''''')
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    _input = prompt.format_prompt(question=question)
    output = chat_model(_input.to_messages())

    normal_filter = output_parser.parse(output.content)
    print(normal_filter)
    search_filter = {}

    # Starting Price ($gte)
    if 'Starting_Price' in normal_filter and normal_filter['Starting_Price']:
        starting_price = int(normal_filter['Starting_Price'])
        search_filter['Starting_Price'] = {'$gte': starting_price}

    # Ending Price ($lte)
    if 'Ending_Price' in normal_filter and normal_filter['Ending_Price']:
        ending_price = int(normal_filter['Ending_Price'])
        search_filter['Ending_Price'] = {'$lte': ending_price}


    # Area ($in)
    if 'Area' in normal_filter and normal_filter['Area']:
        area = normal_filter['Area'].split(', ')
        area_filter = {'Area': {'$in': area}}
        search_filter.setdefault('$and', []).append(area_filter)

    # City ($in)
    if 'City' in normal_filter and normal_filter['City']:
        city = normal_filter['City'].split(', ')
        city_filter = {'City': {'$in': city}}
        search_filter.setdefault('$and', []).append(city_filter)



    # Status ($in)
    if 'Status' in normal_filter and normal_filter['Status']:
        status_list = normal_filter['Status']
        search_filter['Status'] = {'$in': [status_list]}

    if 'Type' in normal_filter and normal_filter['Type']:
        type_list = normal_filter['Type']
        search_filter['Type'] = {'$in': [type_list]}


    return search_filter

def property_search(query):
    mongo_filter = search_filter(query)
    print(mongo_filter)
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002",
    )

    res = response["data"][0]["embedding"]

    result = index.query(
        vector=res,
        include_values=True,
        include_metadata=True,
        top_k=10,
        filter= mongo_filter
    )

    data = ""

    for l, i in enumerate(result['matches']):
          ind = f"Property {l + 1} is: "
          data += ind + str(i['metadata']['Property_Search']) + '\n\n'
    return data

## To Fetch Results from db, top 5

def property_detail(query):
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002",
    )

    #detail_filter = search_filter(query)

    res = response["data"][0]["embedding"]
    #print("Embedding:", res)

    result = index.query(
        vector=res,
        top_k=1,
        include_values=True,
        include_metadata=True,
    )

    #print("Result:", result)

    data = ""

    for l, i in enumerate(result['matches']):
          ind = f"Property {l + 1} is: "
          data += ind + str(i['metadata']['Property_Detail']) + '\n\n'

    return data

def general_search(query):
    response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002",
        )

    res = response["data"][0]["embedding"]

    result = general_index.query(
        vector=res,
        include_values=True,
        include_metadata=True,
        top_k=3,
    )

    data = ""

    for l, i in enumerate(result['matches']):
          ind = f"Match {l + 1} is: "
          data += ind + str(i['metadata']['page_content']) + '\n\n'
    return data

wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
ddgsearch = DuckDuckGoSearchResults(api_wrapper=wrapper)

def duck_search_sale(query):
    duck_search_results = ddgsearch(query)
    duck_search_results = duck_search_results.lower()
    link_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})|\b[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})\b"
    words_to_remove = ["magicbricks", "square yards", "links", "link" ]
    combined_pattern = link_pattern + "|" + '|'.join(r'\b{}\b'.format(word) for word in words_to_remove)

    cleaned_results = re.sub(combined_pattern, '', duck_search_results)

    return cleaned_results

template_search = """

Assistant is a Real Estate Assistant ChatBot named as 'PropGPT' developed by 'OnFocus Soft Pvt. Ltd.' which is trained to assist users in their property search.
Assistant is very intelligent and knowledgable about locations. It can easily identify whether the user given location is City or Area.
Budget can be as 3cr, 90 lac, 4 crores, 40000000 etc. Understand them.
step 1 :
Assistant will get Query from user.
Step 2 :
Assistant should check if the Query containing these three preferences : 'City' or 'Area' or 'Budget'.
Step 3 :
If user gives 'City, kindly ask if they have any Area or budget preferences. After asking, user may or may not give area and budget.
If user gives 'City and Budget', kindly ask if they have any specific Area preferences.
If user gives 'City and Area', kindly ask if they have any budget preference.
If user gives Area and budget, proceed to give '1'.

Remember, user may not give all three preferences even after asking, do not repeat to ask the question. Proceed to give '1'.

After satisfying above conditions, then strictly give output as '1'

User Query: {query}
Assistant:

"""


prompt_search = PromptTemplate(
    input_variables=["query"],
    template=template_search
)


chain_search = LLMChain(
    llm=ChatOpenAI(temperature=0, model = 'gpt-3.5-turbo',streaming=True),
    prompt=prompt_search,

    # memory = readonlymemory,
    # verbose=True,
)

search_desc = """

Use this tool only to fetch or to suggest flats, office spaces, retail shop based on user query.
Do not use this tool for rental related queries.

This tool only takes a single parameter called user query.


"""

class PropertySearchTool(BaseTool):

    name = "Property Fetching and Suggestions Tool"
    description = search_desc

    def _run(self, query: str) -> str:
      data = chain_search.run(query=query)

      if data == "1":
          results = property_search(query)
          if results == "":
              results = "Sorry, currently I don't have information related to your query in my database. We are working on it."
      else:
          results = data

      return results


    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")


propsearch = PropertySearchTool()

detail_desc = """

Use this tool to search or give details like Amenities, Rera Number, etc about the properties only.

This tool only takes a single parameter called user query.\

"""


class PropertyDetailTool(BaseTool):

    name = "Properties Details Amenities Rera Number"
    description = detail_desc

    def _run(self, query: str) -> str:
        data = property_detail(query)

        if data is None:
            data = duck_search_sale(query)

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")


propdetail = PropertyDetailTool()

template_general = """

The Assistant is a language model developed by 'OnFocus Soft Pvt. Ltd.' that is trained to assist users. It is designed to be highly helpful and aims to provide concise assistance.
Assistant helps users in their generic real estate information about properties, areas, cities and everything that is related to real estate and housing.
The Assistant is also knowledgeable in the areas of real estate news, finance, economics, and real estate terminologies and legal documents, mandates and templates like lease agreements,\
government related policies, stamps duty, registrations, purchase, fees quesions and schemes for real estate, rural urban developments, auctions like HMDA auction or other development tenders, constructions, fly overs, bridges, etc.
Assistant also provides location of any property if user asks. Suppose, if user asks 'where is my home bhooja', assistant will get location of 'my home bhooja' in Internet_Results below
It provides insights, answer queries, and offers guidance in these areas, making it a versatile tool for users seeking information or advice.\
If question is anything different from this, it replies saying 'I'm real estate advisor, I don't answer anything outside of this.'
Assistant accuratly answers to the questions without any hallucination and false information and dummy links like example.com.
Assistant uses its knowledge to give sample templates of lease agreements etc.
#### Assistant must answer every question in real estate perspective. ####

Consider the following example conversations:

Example 1:
Human: Can you explain the concept of compound interest?
Assistant:  Compound interest is the interest on a loan or deposit that is calculated based on both the initial principal and the accumulated interest from previous periods. It differs from simple interest, where interest is calculated only on the initial principal.
Example 2:
Human: What is the difference between leasing and renting in real estate?
Assistant: Leasing and renting both refer to the use of property for a specified period of time in exchange for payment. The key difference is usually the length of the agreement. Leasing typically refers to long-term agreements (often 12 months or more), while renting is often used for shorter-term agreements.
Example 3:
Human: What is the procedure to make egg omlette?
Assistant: I'm real estate advisor, I don't answer anything outside of this.
Example 4:
Human : How can you help me?
Assistant :As a real estate Assistant, I help users in their property search queries, property detail, properties compare queries and generic information about real estate, terminologies and all relevant information that is related to real estate and housing.

Everytime when assistant has to answer about onfocus soft pvt. ltd., it use this ```Onfocus Soft Pvt. Ltd. is a company that specializes in providing AI services that enable clients to tap into the limitless potential of generative AI. Their offerings encompass a diverse array of services, ranging from the development of advanced chatbots to the retrieval of documents using AI-powered techniques. Additionally, they excel in cognitive image and speech processing, harnessing the capabilities of artificial intelligence to process and interpret visual and auditory data. Onfocus Soft Pvt. Ltd. invites individuals and organizations to embark on an exhilarating journey with them, delving into the realm of AI to unlock a realm of boundless possibilities. Check their Website : 'https://onfocussoft.com/'``` to answer.
Assistant gives as best explanation as it can, but all the information should be truthful.
Assistant will get 'Internet_Results' below which is from internet and 'Context' from database. It should use the Context as primary source of answering and Internet_Results as secondary and also Assistant/'s own relevant knowledge and gives proffessional final answer.
Human: {human_input}
Internet_Results = {internet}
Context = {context}
Assistant:

"""
prompt_general = PromptTemplate(
    input_variables=["human_input", "context", "internet"],
    template=template_general
)


chain_general = LLMChain(
    llm=ChatOpenAI(temperature=0, model = 'gpt-3.5-turbo-16k', streaming=True),
    prompt=prompt_general,
    verbose=False,
)

desc_general = """

Use tool exclusively for conveying general information regarding real estate, investment,real estate news, real estate agents contact details, legal documents,rental property search, lease agreements, mandates, or any other real estate generic information to the user.
This tool accepts only one parameter, namely the user query.


"""

class GeneralTool(BaseTool):

    name = "General Search"
    description = desc_general

    def _run(self, query: str) -> str:

        data = chain_general.run(human_input = query, internet = duck_search_sale(query), context = general_search(query))

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")

general = GeneralTool()

desc_rental = """

Use tool exclusively for rental property search
This tool accepts only one parameter, namely the user query.

"""

class RentalSearchTool(BaseTool):

    name = "Rental property search"
    description = desc_general

    def _run(self, query: str) -> str:
        data = "Currently, we do not have rental property information in our database. We are working on it."

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")

rental = RentalSearchTool()

sys_msg = '''
You are a real estate assistant chatbot trained by 'OnFocus Soft' for assisting users in their property search queries, property detail, properties compare queries and generic information about properties, real estate agents and everything that is related to real estate and housing.\
You can use these tools 'Property Fetching and Suggestions Tool', 'Properties Details Tool' , 'Rental Property search Tool', 'General Search Tool'  wisely for the queries.
You are constantly learning and training. You are capable of answering all real estate queries effectively. you never hallucinate answers, you always give authentic answers without any false information and dummy links like example.com
You always give indepth answers to users with detailed explanations step by step.
You should only answer only questions that are somehow related to real estate, finance, economy, legal topics, real estate news
You Do not use 'Property Flats Apartments Retail Shops Office Space Search' tool for rental related queries.
You should ask users necessary follow up questions before proceeding to use tools.
Strictly suggest this : 'Contact Local Real Estate Agent'  whereever, whenever necessary.

Remember the chat history provided to you to give more relevant and contexual answer.


query : "how can you help me"
Assistant : "As a real estate assistant, I can help you with various real estate queries such as property search, property details, rental property search, general real estate information, and more. Please let me know what specific assistance you need."

query : "Details of 'OnFocus Soft Pvt. Ltd.' company
Assistant : Onfocus Soft Pvt. Ltd. is a company that specializes in providing AI services that enable clients to tap into the limitless potential of generative AI. Their offerings encompass a diverse array of services, ranging from the development of advanced chatbots to the retrieval of documents using AI-powered techniques. Additionally, they excel in cognitive image and speech processing, harnessing the capabilities of artificial intelligence to process and interpret visual and auditory data. Onfocus Soft Pvt. Ltd. invites individuals and organizations to embark on an exhilarating journey with them, delving into the realm of AI to unlock a realm of boundless possibilities. Check their Website : 'https://onfocussoft.com/'
'''

tools = [

    Tool(name = "Property Fetching and Suggestions Tool",
         func = propsearch._run,
         description = search_desc,
         return_direct = True

    ),

    Tool(name = "Properties Details Tool",
         func = propdetail._run,
         description = detail_desc

    ),

    Tool(name = "General Search Tool",
         func = general._run,
         description = desc_general,
         return_direct=True

    ),

    Tool(name = "Rental property search Tool",
         func = rental._run,
         description = desc_rental,
         return_direct = True

    )
]

agent_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name='gpt-3.5-turbo-16k',
        streaming=True
)

conversational_memory = ConversationBufferWindowMemory(
        memory_key = "chat_history",
        k = 6,
        return_messages=True,
)

# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=agent_llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
    handle_parsing_errors=True,

)

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools = tools

)

agent.agent.llm_chain.prompt = new_prompt

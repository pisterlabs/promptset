import random
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.document_loaders import Docx2txtLoader
from langchain.agents import initialize_agent
from AutoOffer import settings
from langchain.callbacks import get_openai_callback
from AutoOffer.html_manipulation import HTML
from AutoOffer.db import db_funct
import time, schedule, locale


pp = HTML.PropertyProfile()


def create_query(prop): 
    # Define values from property
    offer = locale.format_string("%d", prop[pp.offer_price], grouping=True)
    address = prop[pp.steet_address]
    agent_firstname = prop[pp.agent_first_name]
    public_remarks = prop[pp.public_remarks]

    # Define defualt values
    investor_name = 'Charles Watkins'

    # Select Houston number if location is HOU
    if prop[pp.location] == "HOU":
        investor_number = '832-263-6157'
        investor_location = 'Houston'

    # Select San Antonio number if location is SA
    if prop[pp.location] == "SA":
        investor_number = "210-405-5118"
        investor_location = 'San Antonio'

    # List out the possible intros
    intros = [
    f"Hey {agent_firstname},",
    f"Hey there, {agent_firstname},",
    f"Warm greetings, {agent_firstname},",
    f"Trust you're doing well, {agent_firstname},",
    f"Congratulations on the listing {agent_firstname},",
    f"Delighted to connect with you again, {agent_firstname},",
    f"Hello {agent_firstname},",
    f"Hi there, {agent_firstname},",
    f"Thrilled to be reaching out to you, {agent_firstname},",
    f"Hope you're having a great day, {agent_firstname},",
    f"Sending my best regards, {agent_firstname},",
    f"Hello again, {agent_firstname},",
    f"Hi, {agent_firstname},",
    f"Hope all is well, {agent_firstname},",
    f"Hey {agent_firstname}, hope you're doing great!",
    ]

    # Pick a random intro
    rand_intro = random.choice(intros)
            

    query = f"""
    You are a local {investor_location} investor named {investor_name}, buying properties for investment purposes. 
    Your goal is to write an personable email to real estate agent, {agent_firstname}, to present an offer of ${offer} on the thier MLS listing located at {address}.
    Start the email with "{rand_intro} This is..." then continue with intent of email
    Do not start off mulitple sentences with the same word.


    REALTOR'S REMARKS FROM {agent_firstname}:
    {public_remarks}

    INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
        Using the realtor's remarks craft a sentence detailing positives about the home and why it's a good fit for you. Do not reference the realtor's remarks word for word in the email.
        Jutsify your offfer baecause it gives you the neceassary amount of equity.
        If there are anything negative about the property mentioned in the realtor remarks, use that a justification as well.
        Craft 1 sentence emphasizing cash funding for the deal, a quick close, and little to no closing costs.
        End the email by displaying an eagerness to work together and provide your contact number: {investor_number}.
        DO NOT USE THE WORD "FAIR"
        Keep the email between 300 to 400 words
    """

    return query

def generate_email_body(prop):

    query = create_query(prop)

    # Set API key for OpenAI Service
    openai_api_key = settings.OPENAI_API_KEY

    llm = OpenAI(openai_api_key=openai_api_key,
                temperature=0.7,
                model_name='text-davinci-003')

    # est_tokens = llm.get_num_tokens(query)   
    # print(est_tokens)                   

    # Query the OpenAI database and get the details on the cost
    with get_openai_callback() as cb:
        query_result = llm(query)

        if query_result:

            # Write the created email to the db
            db_funct.multi_db_update(
                mls_id=prop[pp.mls_id],
                data_dict={pp.email_body: query_result,
                           # Store the cost of generating the email
                           pp.ai_cost: cb.total_cost,},
                overwrite=True
            )
        print(cb)

    print(query_result)

def main ():
    # Create a db if there isn't one
    db_funct.create_db()

    # Find all the properties that need an email body made by checking if the Offer_Path is not NULL and the Email_Made is NULL
    props = db_funct.get_sorted_rows_with_null_and_not_null(
                sort_column=pp.last_updated,
                null_list=[
                    pp.offer_sent,
                    pp.email_body
                ],
                not_null_list=[
                    pp.pdf_offer_path,
                ],
            )
    
    # Generate and email body for every selected property
    if props:
        for prop in props:
            generate_email_body(prop)
    else:
        print(f"No properties to create emails for. Will wait for next run")

    print(f'Email body made, waiting for next scheduled run')

if __name__ == '__main__':
    main()

    # This will run the code as soon as it's starting rather than waiting
    schedule.every(10).minutes.do(main)

    while True:
        schedule.run_pending()
        time.sleep(1)
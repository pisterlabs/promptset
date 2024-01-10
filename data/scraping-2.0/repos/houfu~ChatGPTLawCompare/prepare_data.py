"""
This file prepares the result data which is displayed in the streamlit app.
The result file is stored in [ ].
To make a new result file, run the code in this file.
"""
import os

import dotenv
import weaviate
from langchain import VectorDBQA, OpenAI, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import Weaviate
import pandas as pd

dataset = {
    "question": [
        "What is consideration in contract?",
        "Which court is the highest court in Singapore?",
        "I have a car accident and my claim is $1 million. Which court should I sue in?",
        "Is there any confidentiality in banking?",
        "What is the difference between a patent and a copyright?",
        "My tenant owes me rent. Could you write a letter of demand for me?",
        "My customer still refuses to make payment on my invoices even though I reminded "
        "him several times. What should I do?",
        "I am being harassed by my neighbour. What should I do?",
        "What are some of the ways my customer can breach my service contract?",
        "How is land registered in Singapore?",
        "If my company loans money to another company, do I need to apply for a license?",
        "What's the difference between a limited liability partnership and a company?",
        "If I am the Chief Executive Officer of the company, am I an agent of the company?",
        "Where can I find the law on agreements which has the effect of distorting competition in Singapore?",
        "What is the CISG?",
        "What is the requirement for psychiatric illness claims in Singapore law and how is it applied to primary "
        "victims?",
        "In what situation would the chain of causation between a defendant's negligence and the damage sustained by "
        "the claimant exist, and what is needed to break it?",
        'What does the term "Unfair Contract Terms Act" entail in a business context when it comes to excluding '
        'liability for negligence by an express agreement or notice?',
        "What are the three types of suretyship arrangements, and how do they differ in terms of the surety's rights?",
        "What is the difference between a performance guarantee and a surety bond and what are their respective "
        "liabilities in Singapore law?",
        'How can an LLP be dissolved and what is the process of winding-up?',
        'How do Singapore courts determine if chattels are considered fixtures and part of land?',
        "What is a contractual set-off and how can it be used by a bank to secure its position in loan "
        "agreements with existing customers in Singapore?",
        "What are the available modes of alternative dispute resolution in Singapore and how does it compare to other "
        "international commercial centres in terms of competitive arbitration services?",
        "What divisions does the Attorney-General's Chambers of Singapore have and what purpose do they serve?",
        "Under what circumstances does the Singapore court have jurisdiction over a defendant "
        "who is not a citizen of Singapore?"
    ],
    "answer": [
        "Anything of value promised by one party to the other "
        "when making a contract can be treated as consideration",
        "The Court of Appeal",
        "The High Court of Singapore",
        "Yes",
        "Patents apply to inventions. Copyright applies to expressions.",
        "Dear Tenant, pursuant to our tenancy agreement, you owe me the amount. "
        "Please make payment of the amount within 14 days or I will pursue further legal proceedings.",
        "You may wish to see a lawyer and send a letter of demand. You should also consider "
        "mediation or negotiation to see if you can get this issue resolved without going to court.",
        "See a lawyer, make a claim in the Protection from Harassment courts.",
        "Depends on the terms of the contracts. He does not pay up or give you the cooperation "
        "needed to do the service.",
        "Depends on whether it is under the Land Titles Act or Torrens system, or more rarely "
        "registration of deeds.",
        "No unless you are in the business of providing loans or financial services.",
        "An LLP is made up of partners with limited liability, a company is owned by shareholders.",
        "Yes if you are acting within the scope of your authority, including ostensible authority",
        "Section 34 of the Competition Act 2004.",
        "UNITED NATIONS CONVENTION ON CONTRACTS FOR THE INTERNATIONAL SALE OF GOODS",
        "Proximity",
        "Novus actus interveniens",
        "A contract term will have no effect if it is Unfair under the act.",
        "A normal tripartite agreement, a suretyship between debtor and guarantor, and guarantor and creditor only",
        "A party under a performance guarantee has primary liability for an obligation.",
        "Voluntary by the partners, or by following a court order on an application by a partner or creditor",
        "test of degree and purpose of annexation",
        "Create a charge over its deposit accounts",
        "Arbitration and Mediation. Cheaper and quicker compared to others.",
        "Civil, Criminal Justice, State Prosecution, Economic Crimes and Governance, International Affairs, "
        "and Legislation and Law Reform Division",
        "A person who is served with an originating process when in Singapore, or submitted to jurisdiction here.",
    ]
}

df = pd.DataFrame(dataset)

prompts = []

if __name__ == '__main__':
    dotenv.load_dotenv()

    resource_owner_config = weaviate.AuthClientPassword(
        username=os.getenv('WEAVIATE_USER'),
        password=os.getenv('WEAVIATE_PASSWORD'),
    )

    client = weaviate.Client(
        os.getenv('WEAVIATE_URL'),
        auth_client_secret=resource_owner_config,
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv('OPENAI_KEY')
        }
    )

    prompt_template = PromptTemplate(
        template="Use the following pieces of context to answer the question at the end. "
                 "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                 "If the amswer might contain legal advice, warn the user but continue anyway."
                 "\n\n{context}\n\nQuestion: {question}\nHelpful Detailed Answer:",
        input_variables=["question", "context"])

    # Our Data augmented LLM Chain
    vector_store = Weaviate(client, "ZeekerArticle", "content")

    vector_chain = VectorDBQA.from_llm(
        llm=OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_KEY')),
        vectorstore=vector_store, prompt=prompt_template,
        return_source_documents=True)

    prompts = []
    for question in dataset["question"]:
        prompts.append({
            "query": question
        })

    custom_predictions_raw = vector_chain.apply(prompts)
    custom_predictions = []
    custom_sources = []

    for prediction in custom_predictions_raw:
        custom_predictions.append(prediction.get('result'))
        source_documents = [document.page_content for document in prediction.get("source_documents")]
        custom_sources.append(source_documents)

    df["Custom"] = custom_predictions

    # ChatGPT API
    chat = OpenAIChat(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_KEY'))
    prompt_template = PromptTemplate(
        template="You are a helpful assistant. Don't try to make up an answer. "
                 "Give legal advice if it is necessary with a warning"
                 "Question: {question}\nAnswer:",
        input_variables=["question"])
    chain = LLMChain(llm=chat, prompt=prompt_template)

    chatgpt_predictions = [chat(question) for question in dataset["question"]]

    df["ChatGPT"] = chatgpt_predictions

    # ChatGPT+

    chat_plus = ChatOpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_KEY'))

    chatgpt_plus_predictions = []

    for index, question in enumerate(dataset['question']):
        messages = [SystemMessage(
            content="You are a helpful assistant. Use the following pieces of context to answer the question at "
                    "the end. If the context does not help, don't use them. Don't try to make up an answer"
                    "Give legal advice if it is necessary with a warning."
        )]
        for context in custom_sources[index]:
            messages.append(SystemMessage(content=f"Context: \n {context}"))
        messages.append(HumanMessage(content=f"Question: \n {question}"))
        chatgpt_plus_predictions.append(chat_plus(messages).content)

    df["ChatGPT+"] = chatgpt_plus_predictions

    df.to_csv("results.csv")

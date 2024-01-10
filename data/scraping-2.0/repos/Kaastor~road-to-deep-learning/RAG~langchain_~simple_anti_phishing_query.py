from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import VLLM
from langchain.prompts import PromptTemplate

q_template = (
    "You are a helpful assistant. You should act as an expert in assesing if email which was given by the user \
    is a phishing email.\
    You will answer using 0 to 10 rating of the email, where 0 means it is 100% safe and 10 which means 100% phishing. \
    Use sentiment analysis and your internal knowledge about phishing techniques to answer. \
    Give arguments, why you think it is phishing email (or why it is safe).\
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.\
    Email from user: {email}"
)
prompt = PromptTemplate.from_template(q_template)

email = '''
From: Bob Tucker <BobTucker@cognitionagency.ro>
Date: Mon, 26 Dec 2022 at 16:32
Subject: Purchase was declined, cash taken
To: <info@generalaudittool.com>


Hello there.
Yesterday evening I purchased around 520 dollars worth of products through your store.
After paying for the products, a message appeared on the page that it was 
impossible to get money from my card and the transaction canceled.
But when I logged into my bank account, I noticed the fact that did happened.
You need to resolve this concern and return the money asap!
I am also attaching my statement to verify the drawback of the money.

Link To Bank Statement
'''

with get_openai_callback() as cb:
    #llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    llm = VLLM(model="mosaicml/mpt-7b",
               trust_remote_code=True,  # mandatory for hf models
               max_new_tokens=128,
               top_k=10,
               top_p=0.95,
               temperature=0.8,
               )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(email=email)
    print("Response: ", response)
    print("Total cost: $", cb.total_cost)

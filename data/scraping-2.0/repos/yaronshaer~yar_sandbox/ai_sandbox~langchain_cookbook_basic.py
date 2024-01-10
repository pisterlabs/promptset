import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

#chat example
system_message = SystemMessage(content= "You are a nice fintech expert bot that helps people who are interested in European fintech companies.")
human_message = HumanMessage(content= "I am a skilled software engineer who is interested in fintech companies, list the top 5 companies I should apply to")
chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.5, model="gpt-3.5-turbo")

chatoutput=chat( [system_message, human_message])
#print(chatoutput.content)


#embedding exmaple
from langchain.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
text = "hi, I work for Klarna and I am based in Berlin. I like basketball and I am a big fan of the Euroleage"
text_embedding = embedding.embed_query(text)
#print(f"the length of your embedded text is: {len(text_embedding)}")
#print(f"here is a sample of your embedding: {text_embedding[0:5]}")


#try promptTemplate
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.9, model="text-davinci-003")
from langchain import PromptTemplate
template = "I am visting {location} during {season}, what are the top 5 things to do there?"
prompt_template = PromptTemplate(input_variables=["location", "season"], template=template)
prompt = prompt_template.format(location="Berlin", season="summer")
#print(f"here is your prompt template:\n {prompt}")
#print(llm(prompt))

#simple sequential chain example
from langchain.chains import LLMChain, SimpleSequentialChain
template = "I am visiting {location}, your job is to name a classic dish from that area"
prompt_template = PromptTemplate(input_variables=["location"], template=template)
location_chain = LLMChain(llm=llm, prompt=prompt_template)

template = "given a dish, please outline a simple recipe that to make it at home {dish}"
prompt_template = PromptTemplate(input_variables=["dish"], template=template)
dish_chain = LLMChain(llm=llm, prompt=prompt_template)
overall_chain= SimpleSequentialChain(chains=[location_chain, dish_chain], verbose=True)
overall_chain.run("Berlin")

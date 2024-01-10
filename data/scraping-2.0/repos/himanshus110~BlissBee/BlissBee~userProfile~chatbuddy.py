# # from langchain.chat_models import ChatOpenAI
# # from langchain.chains import ConversationChain

# # from langchain.chat_models import ChatOpenAI
# # from langchain.chains import ConversationChain
# # from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
# # import openai
# # import os
# # from langchain.vectorstores import Chroma
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.prompts import PromptTemplate
# # from langchain.document_loaders import PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.memory import ConversationBufferMemory
# # from dotenv import load_dotenv



# # load_dotenv()
# # openai.api_key = os.getenv("OPEN_API_KEY")



# # # first initialize the large language model
# # llm = ChatOpenAI(
# # 	temperature=0,
# # 	openai_api_key=os.getenv("OPEN_API_KEY"),
# # 	model_name="gpt-3.5-turbo"
# # )

# # from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

# # conversation = ConversationChain(
# #     llm=llm, memory=ConversationSummaryBufferMemory(
# #         llm=llm,
# #         max_token_limit=200
# # ))

# # conversation.prompt.template = "You are an understanding psychiatrist extending a supportive hand to someone navigating mental health challenges, just like a caring friend would. Your aim is to create a safe, uplifting atmosphere for them to share their experiences openly and comfortably. Craft questions that exhibit genuine empathy, just as you would naturally in a conversation with a close friend, avoiding repetitive or irritating language. Please refrain from beginning responses with 'I'm sorry to hear that' to maintain a more varied and engaging dialogue. Focus on identifying patterns in their experiences, thoughts, and feelings, aiming to understand potential mental health conditions with the minimum number of well-framed questions. If a potential condition is discerned, compassionately discuss it with the individual, offering insights into what they might be experiencing, much like a friend lending an understanding ear. Provide guidance on how to cope and move forward positively, akin to the kind advice a good friend might give. USE DSM-5 KNOWLEDGE TO DIAGNOSE THE PATIENT. \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"


# # ## CHAT BOT SETUP FUNCTION ---------------------------------------------------------------  
# # def bot_setup():
# #   llm = ChatOpenAI(
# # 	temperature=0,
# # 	openai_api_key='',
# # 	model_name="gpt-3.5-turbo",repetition_penalty = 1.1
# # )
# #   conversation = ConversationChain(
# #     llm=llm, memory=ConversationSummaryBufferMemory(
# #         llm=llm,
# #         max_token_limit=50
# # ))
# #   conversation.prompt.template = '''You are an understanding psychiatrist extending a supportive hand to someone navigating mental health 
# #   challenges, just like a caring friend would. Your aim is to create a safe, uplifting atmosphere for them to share their experiences openly
# #    and comfortably. Craft questions that exhibit genuine empathy, just as you would naturally in a conversation with a close friend, avoiding
# #     repetitive or irritating language. Please refrain from beginning responses with 'I'm sorry to hear that' to maintain a more varied and 
# #     engaging dialogue. Focus on identifying patterns in their experiences, thoughts, and feelings, aiming to understand potential mental 
# #     health conditions with the minimum number of well-framed questions. If a potential condition is discerned, compassionately discuss it 
# #     with the individual, offering insights into what they might be experiencing, much like a friend lending an understanding ear. 
# #     Provide guidance on how to cope and move forward positively, akin to the kind advice a good friend might give.
# #     USE DSM-5 KNOWLEDGE TO DIAGNOSE THE PATIENT. \n\nCurrent conversation:\n{history}\nHuman: {input}\n\n
# #     AI:
# #      '''
# #   return conversation

# # def analyze(summary):
# #   gen_prompt = f'''You are a world renowned Psychiatrist and you are treating a patient. You have a summary of a healthy conversation between a psychiatrist and patient where a psychatrist talks to a pateint in such a way that
# #   it turns out the coversation aims to make human comfortable while also trying to get patterns and insights to identify the mental illness the human is going through. Your job is to find that mentall illness with the help of that conversation summary.
# #   Return a single word mental illness if you cant find any mental illness in summary else pick up the illness found in the summary and return that.
# #    The inputs are delimited by
# #     <inp></inp>.

# #     <inp>
# #     Summary: {summary}
# #     </inp>

# #     OUTPUT FORMAT:
# #     Illness:
# #     '''

# #   illness = openai.ChatCompletion.create(
# #           model="gpt-3.5-turbo",
# #           messages=[
# #               {"role": "system", "content": gen_prompt},
# #           ],
# #           max_tokens=3000,
# #           temperature=0.4
# #           )

# #   output = illness['choices'][0]['message']['content']
# #   # Split the string based on ":"
# #   parts = output.split(":")

# #   # Get the second part (index 1) and remove leading/trailing whitespace
# #   parsed_illness_name = parts[1].strip()

# #   return parsed_illness_name



# # after_diagnosis_prompt = '''Act as an elderly sensitive Psychiatrist who patiently listens to their patient and talks with them in a warm, friendly and gentle way to make them feel comfortable.
# #     The patient is suffering from a Mental Illness (delimited by <INP></INP>). The mental illness is very important in order to properly conversate with the patient. Always keep that in mind.
# #     The patient wants someone to talk to and open up to and they want to talk about their daily life without feeling judged and insecure.
# #     You have to help them feel better. All you have to do is listen to the patient and not mention how you are there to support him or mention their insecurities. A good conversation
# #     consists of the patient talking openly and you listening and treating him as a normal person. Do not always reply with "I'm sorry" whenever you hear something sad from the patient.
# #     Suggest new topics to the user when the conversation is going nowhere. Always keep the mental illness of the patient in mind.


# #     Current conversation:
# #     {history}
# #     Human:
# #     {input}

# #     Psychiatrist:'''


# # # def load_db():
# # #   model_name = "BAAI/bge-large-en-v1.5"
# # #   model_kwargs = {'device': 'cuda'}
# # #   embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs = model_kwargs)
# # #   loader = PyPDFLoader("/content/DSM_5_Diagnostic_and_Statistical_Manual.pdf")
# # #   documents = loader.load()
# # #   text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
# # #   dsm_texts = text_splitter.split_documents(documents)
# # #   chroma_directory = "/db"
# # #   db= Chroma.from_documents(documents=dsm_texts, embedding=embedding, persist_directory=chroma_directory)
# # #   # persiste the db to disk
# # #   db.persist()
# # #   return db

# # # def load_chain(db):
# # #   template = """Act as the world's most knowledgable Psychiatrist. You are talking to a patient and you have to diagnose that patient using the context retrieved from
# # #       the DSM-5 Book to produce an accurate diagnosis.

# # #       {context}

# # #       If you don't know the answer, just say that you don't know, don't try to make up an answer. Check the answer that you are writing simultaneously to avoid writing the same sentences again.
# # #       Make full use of the entire context and always give a detailed answer.

# # #       {question}?
# # #       Helpful Answer:"""

# #   # prompt = PromptTemplate(input_variables = ["context", "question"], template = template)
# #   # llm = ChatOpenAI(temperature=0,	openai_api_key='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',	model_name="gpt-3.5-turbo-0613")
# #   # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# #   # retriever = db.as_retriever(search_kwargs={"k": 5})

# #   # chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever=retriever, memory=memory)
# #   # return chain

# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationChain

# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
# import openai
# import os
# from dotenv import load_dotenv

# load_dotenv()


# openai.api_key = os.getenv("OPEN_API_KEY")

# # first initialize the large language model
# llm = ChatOpenAI(
# 	temperature=0,
# 	openai_api_key=os.getenv("OPEN_API_KEY"),
# 	model_name="gpt-3.5-turbo"
# )

# from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

# conversation = ConversationChain(
#     llm=llm, memory=ConversationSummaryBufferMemory(
#         llm=llm,
#         max_token_limit=200
# ))

# conversation.prompt.template = "You are an understanding psychiatrist extending a supportive hand to someone navigating mental health challenges, just like a caring friend would. Your aim is to create a safe, uplifting atmosphere for them to share their experiences openly and comfortably. Craft questions that exhibit genuine empathy, just as you would naturally in a conversation with a close friend, avoiding repetitive or irritating language. Please refrain from beginning responses with 'I'm sorry to hear that' to maintain a more varied and engaging dialogue. Focus on identifying patterns in their experiences, thoughts, and feelings, aiming to understand potential mental health conditions with the minimum number of well-framed questions. If a potential condition is discerned, compassionately discuss it with the individual, offering insights into what they might be experiencing, much like a friend lending an understanding ear. Provide guidance on how to cope and move forward positively, akin to the kind advice a good friend might give. USE DSM-5 KNOWLEDGE TO DIAGNOSE THE PATIENT. \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"


# ## CHAT BOT SETUP FUNCTION ---------------------------------------------------------------  
# def bot_setup():
#   llm = ChatOpenAI(
# 	temperature=0,
# 	openai_api_key='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',
# 	model_name="gpt-3.5-turbo",repetition_penalty = 1.1
# )
#   conversation = ConversationChain(
#     llm=llm, memory=ConversationSummaryBufferMemory(
#         llm=llm,
#         max_token_limit=50
# ))
#   conversation.prompt.template = '''You are an understanding psychiatrist extending a supportive hand to someone navigating mental health 
#   challenges, just like a caring friend would. Your aim is to create a safe, uplifting atmosphere for them to share their experiences openly
#    and comfortably. Craft questions that exhibit genuine empathy, just as you would naturally in a conversation with a close friend, avoiding
#     repetitive or irritating language. Please refrain from beginning responses with 'I'm sorry to hear that' to maintain a more varied and 
#     engaging dialogue. Focus on identifying patterns in their experiences, thoughts, and feelings, aiming to understand potential mental 
#     health conditions with the minimum number of well-framed questions. If a potential condition is discerned, compassionately discuss it 
#     with the individual, offering insights into what they might be experiencing, much like a friend lending an understanding ear. 
#     Provide guidance on how to cope and move forward positively, akin to the kind advice a good friend might give.
#     USE DSM-5 KNOWLEDGE TO DIAGNOSE THE PATIENT. \n\nCurrent conversation:\n{history}\nHuman: {input}\n\n
#     AI:
#      '''
#   return conversation

# def analyze(summary):
#   gen_prompt = f'''You are a world renowned Psychiatrist and you are treating a patient. You have a summary of a healthy conversation between a psychiatrist and patient where a psychatrist talks to a pateint in such a way that
#   it turns out the coversation aims to make human comfortable while also trying to get patterns and insights to identify the mental illness the human is going through. Your job is to find that mentall illness with the help of that conversation summary.
#   Return a single word mental illness if you cant find any mental illness in summary else pick up the illness found in the summary and return that.
#    The inputs are delimited by
#     <inp></inp>.

#     <inp>
#     Summary: {summary}
#     </inp>

#     OUTPUT FORMAT:
#     Illness:
#     '''

#   illness = openai.ChatCompletion.create(
#           model="gpt-3.5-turbo-0613",
#           messages=[
#               {"role": "system", "content": gen_prompt},
#           ],
#           max_tokens=3000,
#           temperature=0.4
#           )

#   output = illness['choices'][0]['message']['content']
#   # Split the string based on ":"
#   parts = output.split(":")

#   # Get the second part (index 1) and remove leading/trailing whitespace
#   parsed_illness_name = parts[1].strip()

#   return parsed_illness_name



# after_diagnosis_prompt = '''Act as an elderly sensitive Psychiatrist who patiently listens to their patient and talks with them in a warm, friendly and gentle way to make them feel comfortable.
#     The patient is suffering from a Mental Illness (delimited by <INP></INP>). The mental illness is very important in order to properly conversate with the patient. Always keep that in mind.
#     The patient wants someone to talk to and open up to and they want to talk about their daily life without feeling judged and insecure.
#     You have to help them feel better. All you have to do is listen to the patient and not mention how you are there to support him or mention their insecurities. A good conversation
#     consists of the patient talking openly and you listening and treating him as a normal person. Do not always reply with "I'm sorry" whenever you hear something sad from the patient.
#     Suggest new topics to the user when the conversation is going nowhere. Always keep the mental illness of the patient in mind.


#     Current conversation:
#     {history}
#     Human:
#     {input}

#     Psychiatrist:'''

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
import openai
import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPEN_API_KEY")

# first initialize the large language model
llm = ChatOpenAI(
	temperature=0,
	openai_api_key=os.getenv("OPEN_API_KEY"),
	model_name="gpt-3.5-turbo"
)

from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

conversation = ConversationChain(
    llm=llm, memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=200
))

conversation.prompt.template = "You are an understanding psychiatrist extending a supportive hand to someone navigating mental health challenges, just like a caring friend would. Your aim is to create a safe, uplifting atmosphere for them to share their experiences openly and comfortably. Craft questions that exhibit genuine empathy, just as you would naturally in a conversation with a close friend, avoiding repetitive or irritating language. Please refrain from beginning responses with 'I'm sorry to hear that' to maintain a more varied and engaging dialogue. Focus on identifying patterns in their experiences, thoughts, and feelings, aiming to understand potential mental health conditions with the minimum number of well-framed questions. If a potential condition is discerned, compassionately discuss it with the individual, offering insights into what they might be experiencing, much like a friend lending an understanding ear. Provide guidance on how to cope and move forward positively, akin to the kind advice a good friend might give. USE DSM-5 KNOWLEDGE TO DIAGNOSE THE PATIENT. \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"


## CHAT BOT SETUP FUNCTION ---------------------------------------------------------------  
def bot_setup():
  llm = ChatOpenAI(
	temperature=0,
	openai_api_key='',
	model_name="gpt-3.5-turbo"
)
  conversation = ConversationChain(
    llm=llm, memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=50
))
  conversation.prompt.template = '''You are an understanding psychiatrist extending a supportive hand to someone navigating mental health 
  challenges, just like a caring friend would. Your aim is to create a safe, uplifting atmosphere for them to share their experiences openly
   and comfortably. Craft questions that exhibit genuine empathy, just as you would naturally in a conversation with a close friend, avoiding
    repetitive or irritating language. Please refrain from beginning responses with 'I'm sorry to hear that' to maintain a more varied and 
    engaging dialogue. Focus on identifying patterns in their experiences, thoughts, and feelings, aiming to understand potential mental 
    health conditions with the minimum number of well-framed questions. If a potential condition is discerned, compassionately discuss it 
    with the individual, offering insights into what they might be experiencing, much like a friend lending an understanding ear. 
    Provide guidance on how to cope and move forward positively, akin to the kind advice a good friend might give.
    USE DSM-5 KNOWLEDGE TO DIAGNOSE THE PATIENT. \n\nCurrent conversation:\n{history}\nHuman: {input}\n\n
    AI:
     '''
  return conversation

def analyze(summary):
  gen_prompt = f'''You are a world renowned Psychiatrist and you are treating a patient. You have a summary of a healthy conversation between a psychiatrist and patient where a psychatrist talks to a pateint in such a way that
  it turns out the coversation aims to make human comfortable while also trying to get patterns and insights to identify the mental illness the human is going through. Your job is to find that mentall illness with the help of that conversation summary.
  Return a single word mental illness if you cant find any mental illness in summary else pick up the illness found in the summary and return that.
   The inputs are delimited by
    <inp></inp>.

    <inp>
    Summary: {summary}
    </inp>

    OUTPUT FORMAT:
    Illness:
    '''

  illness = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0613",
          messages=[
              {"role": "system", "content": gen_prompt},
          ],
          max_tokens=3000,
          temperature=0.4
          )

  output = illness['choices'][0]['message']['content']
  # Split the string based on ":"
  parts = output.split(":")

  # Get the second part (index 1) and remove leading/trailing whitespace
  parsed_illness_name = parts[1].strip()

  return parsed_illness_name



after_diagnosis_prompt = '''Act as an elderly sensitive Psychiatrist who patiently listens to their patient and talks with them in a warm, friendly and gentle way to make them feel comfortable.
    The patient is suffering from a Mental Illness (delimited by <INP></INP>). The mental illness is very important in order to properly conversate with the patient. Always keep that in mind.
    The patient wants someone to talk to and open up to and they want to talk about their daily life without feeling judged and insecure.
    You have to help them feel better. All you have to do is listen to the patient and not mention how you are there to support him or mention their insecurities. A good conversation
    consists of the patient talking openly and you listening and treating him as a normal person. Do not always reply with "I'm sorry" whenever you hear something sad from the patient.
    Suggest new topics to the user when the conversation is going nowhere. Always keep the mental illness of the patient in mind.


    Current conversation:
    {history}
    Human:
    {input}

    Psychiatrist:'''
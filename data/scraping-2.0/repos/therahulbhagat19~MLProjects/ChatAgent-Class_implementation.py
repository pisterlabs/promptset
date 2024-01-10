import os
from langchain import LLMMathChain, PromptTemplate,LLMChain
from langchain.llms import OpenAI
from langchain.tools import YouTubeSearchTool
from langchain.agents import load_tools
from langchain.agents import initialize_agent,Tool,AgentExecutor
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain import SerpAPIWrapper
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings, GooglePalmEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from langchain.llms import GooglePalm
from langchain.chat_models import ChatGooglePalm
from langchain.agents import ConversationalChatAgent,ConversationalAgent
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

#new_parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(), llm=ChatOpenAI())



os.environ['GOOGLE_API_KEY'] = 'AIzaSyCVi7tD4ekch-AGSILgyIz19Xi8F-aWjAg'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


#from langchain import HuggingFaceHub

#repo_id = "google/flan-t5-xl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options


from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch


TEMPLATE  =  """You are a friendly chatbot, having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot: """



#from getpass import getpass
#HUGGINGFACE_API_KEY = getpass()
#os.environ["HUGGINGFACE_API_KEY"] = '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'
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


os.environ["OPENAI_API_KEY"] = "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"
#os.environ["SERPAPI_API_KEY"] = "c2649e2ecd9819a341eb811a1b308356b5aae3141e1a7a1077130ac9b3ce5563"

# Base Chat model class


class ChatAgent:
    
    def __init__(self,memory,model = 'gpt3.5'):
        match model:
            case 'gpt3.5':
                self.llm = ChatOpenAI(temperature=0.2)
            case 'gpt4':
                self.llm = ChatOpenAI(model_name = 'gpt-4',temperature=0.2)
            case 'palm':
                self.llm = GooglePalm(temperature=0.2)
            case 'palmchat':
                self.llm = ChatGooglePalm(temperature=0.2)
        self.tools = load_tools(['pal-math'], llm= self.llm)
        self.tools.append(DuckDuckGoSearchRun())
        self.llm_math_chain = LLMMathChain.from_llm(llm=self.llm, verbose=True)
        self.llm_math = Tool(
        name="Calculator",
        func=self.llm_math_chain.run,
        description="useful for when you need to answer questions about mathematics and do calculations"
        )
        self.tools.append(YouTubeSearchTool())
        self.prompt  = PromptTemplate(input_variables=["chat_history", "human_input"], template=TEMPLATE)
        self.tools.append(self.llm_math)
        self.count=0
        self.cost=0
        self.memory =  memory
        if(model in ['palm']):
            self.agent = ConversationalAgent.from_llm_and_tools(llm=self.llm, tools=self.tools,
                                                                    verbose=True,memory=self.memory)

            self.agent_chain = AgentExecutor.from_agent_and_tools(agent=self.agent,
                                                                tools=self.tools,
                                                                memory=self.memory,
                                                                verbose=True)
            self.llm_chain = LLMChain(
                                        llm=self.llm,
                                        prompt=self.prompt,
                                        verbose=True,
                                        memory=self.memory,
                                    )
            
        else:
            self.agent_chain = initialize_agent(self.tools, self.llm, agent= AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                                          memory = self.memory)
            self.llm_chain = LLMChain(
                                        llm=self.llm,
                                        prompt=self.prompt,
                                        verbose=True,
                                        memory=self.memory,
                                    )

    def chat(self,query):
        
        #print(output)
        with get_openai_callback() as cb:
            try:
                    output = self.agent_chain.run(query)
                    self.count += int(cb.total_tokens)
                    self.cost += cb.total_cost
                    print(output,f'Tokens used : {self.count}  Cost: {self.cost}')
                    return {output,self.count}
                
            except :
                print('Entering the except block:')
                try:
                    output = self.llm_chain.predict(human_input=query)
                    self.count += int(cb.total_tokens)
                    print(output,f'Tokens used : {self.count}')
                    return {output,self.count}
                except:
                    output = 'Please try again with a better prompt.'
                
                    print(output)
                
    
                
                    

            


#Image captioning class

class ImageCaption:
    def __init__(self):
    #model and processor from downloaded checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor=AutoProcessor.from_pretrained('/workspace/image_caption/processor')
        self.model = AutoModelForCausalLM.from_pretrained('/workspace/image_caption/model').to(self.device)
    
    def image_caption(self,img):
        
        def generate_caps(imge) :   
            inputs = self.processor(images=imge, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values
            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=100)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_caption
        
        context = generate_caps(img)
        #prompt_template = PromptTemplate(input_variables=["context","query"], template=template)
        
        
        return context


#PDF chatbot class
class PDFChatBot:
    def __init__(self,pdf_path):
        self.loader = PyPDFLoader(pdf_path)
        self.pages = self.loader.load_and_split()
        self.embeddings = OpenAIEmbeddings()
        self.vectordbP = Chroma.from_documents(self.pages, embedding=self.embeddings, persist_directory="./pdf/")
        self.vectordbP.persist()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.4) , self.vectordbP.as_retriever(), memory=self.memory)
        self.chat_model = ChatAgent(self.memory)

        
    def ask_on_pdf(self,query):
        outputs = self.pdf_qa.run(query)
        print(outputs)
        final = self.chat_model.agent_chain.run(f'With this additional information {outputs}, answer the following question: {query}')
        
        return final    

#Persona Chat bot class

class PersonaChatBot:
    def __init__(self,pdf_path,name):
        self.loader = PyPDFLoader(pdf_path)
        self.pages = self.loader.load_and_split()
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.pages, embedding=self.embeddings, persist_directory=f"./persona/{name}/")
        #self.vectordb.persist()
        self.name = name
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.6) , self.vectordb.as_retriever(), memory=self.memory)
        self.persona = self.pdf_qa.run(f""" Highlight their experiences, achievements, and notable events in their life. Explore their personality traits, including their strengths, weaknesses, values, and attitudes. Discuss their interactions with others, their approach to challenges, and their overall character. Analyze their decision-making style, communication skills, and emotional intelligence. Provide insights into their passions, interests, and motivations. Delve into their relationships, both personal and professional, and how they navigate various social dynamics. Lastly, conclude with an assessment of their overall character, highlighting their unique qualities and contributions. While doing this try to keep it short""")
        print('Persona made!!! Ask away:')   
       # self.llm_chain = LLMChain(
        #                                llm=OpenAI(),
         #                               prompt=self.prompt,
          #                              verbose=True,
           #                             memory=self.memory,
            #                        )
        
    def ask_on_pdf(self,query):
        try:
            print('path 1')
            outputs = self.pdf_qa.run(f"""Talk like you are {self.name}  and use words like I and me as you are pretending to be {self.name} 
            and your brief introduction is {self.persona}. Use verbs like I and me explicitly and answer in short: {query}""")
            print(outputs)
            return outputs
            
        except:
            print('2')

            

#Which class to use

print("Enter 1 for Image chat bot:\n\nEnter 2 to chat with a Persona ChatBot\n\nEnter 3 for a personal assitant\n\nEnter 4 for a PDF Chat bot\n\n")
inn = input()

if(inn=='2'):
    #path = '../image_caption/images.jpeg'
    path = 'Virtual GF.pdf'
    chat = PersonaChatBot(path,name = 'Suhani')
    #chat.ask_on_pdf()
    #image = Image.open(path)
    
    
    while 1:
        query = input()
        if(query!='-1'):
            chat.ask_on_pdf(query)
        else:
            break
elif(inn=='1'):
    imgcap = ImageCaption()
    path = '../image_caption/images.jpeg'
    image = Image.open(path)
    context = imgcap.image_caption(image)
    print(context)
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    chat = ChatAgent(memory)
    while 1:
        query = input()
        if(query!='-1'):
            chat.chat(f'With reference to this additional information given about the image\n{context}, answer the question{query}')
        else:
            break

elif(inn=='3'):
    memory = ConversationBufferWindowMemory(memory_key="chat_history",return_messages=True)
    #memory = ConversationSummaryMemory(llm=OpenAI(temperature=0.5),memory_key="chat_history",return_messages=True)
    print('Enter Model name to use: \ngpt 4  gpt3.5   palm   palmchat\n')
    model = input()
    chat = ChatAgent(memory,model)
    while 1:
        query = input()
        if(query!='-1'):
            response = chat.chat(query)
            print('response')
            print(response)
        else:
            
            break
elif(inn=='4'):
    path = 'Virtual GF.pdf'
    chat = PDFChatBot(path)
    #chat.ask_on_pdf()
    #image = Image.open(path)
    
    
    while 1:
        query = input()
        if(query!='-1'):
            chat.ask_on_pdf(query)
        else:
            break    

from config import All_keys
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import pinecone
from langchain.prompts import PromptTemplate
import streamlit as st
from streamlit_option_menu import option_menu
import threading
import tempfile
import os
          
          
def main_page():
    print("execuation")
    
    model=ChatOpenAI(openai_api_key=All_keys.Openai_API_KEY,temperature=0.2,model='gpt-4',max_tokens=2000)# add further parameters as per user preference
    embeddings = OpenAIEmbeddings(openai_api_key=All_keys.Openai_API_KEY)
    PINECONE_API_KEY = All_keys.PINECONE_API_KEY
    PINECONE_API_ENV = All_keys.PINECONE_API_ENV


    if 'book' not in st.session_state:
        st.session_state.book = []
    if 'all_personas' not in st.session_state:
        st.session_state.all_personas = []

    # book=[] #creation of main book
    # st.session_state.all_personas=[] #all personas stored here

    chatmemory = ConversationBufferWindowMemory(k=3)

    stop_execution=True

    class Bot:
        def __init__(self,**kwargs) -> None:
            self.name=kwargs['name']
            self.characteristics=kwargs['characteristics']
            self.interests=kwargs['interests']
            self.knowledge_file=kwargs['knowledge_file']

        def knowledge(self,kpath):
            loader=TextLoader(file_path=kpath)
            data=loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            texts = text_splitter.split_documents(data)
            # initialize pinecone
            pinecone.init(
                api_key=PINECONE_API_KEY,  # find at app.pinecone.io
                environment=PINECONE_API_ENV  # next to api key in console
            )
            index = "bookbots" # put in the name of pinecone index
            docsearch = Pinecone.from_documents(texts, embeddings, index_name=index)
            return docsearch
        
        def generate_prompt(self,other_model_input:str)->None:
            global chatmemory
            if len(other_model_input)==0:
                other_model_input='Start a story for a book'

            query=f"""
                    Previous line: "{other_model_input}"      
                    Interests: {self.interests}. 
                    Characteristics: {self.characteristics}.
                    """
            
            sysmsg=PromptTemplate.from_template("""
                                            %INSTRUCTIONS:
                    Suppose that event has occured as follows defined in "Previous line".
                    After that tell us what is likely to occur next from the information given.         
                    Write this one sentence such that its tone matches a persona with Interests defined in "Interests"
                    Write this one sentence such that its tone matches a persona characteristics as defined in "Characteristics"
                    Write this one sentence such that a book is being written.
                %RESTRICTIONS:
                    IMPORTANT :: Aim strictly for one easy short sentence with subject object and verb  
                                            """)
            sysmsg.format()
            qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=self.knowledge(self.knowledge_file).as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            chain_type="stuff",memory=chatmemory,prompt=sysmsg)
            response=qa_chain.run({"query":query})
            return response

    def init_conversation2(bot1:Bot,bot2:Bot,res):
        global stop_execution,book
        if stop_execution:  
            res=(f"{bot2.generate_prompt(res).split('.')[0]}.")
            st.session_state.book.append(f"{res}")
            st.text_area('PERSONA1',value=f"Persona {bot2.name} : {res}")
            print('\n'.join(book[-1:]))
            init_conversation1(bot1,bot2,res)
        else:
            st.error("Process stopped")

    def init_conversation1(bot1:Bot,bot2:Bot,res):
        global stop_execution,book
        if stop_execution:        
            res= (f"{bot1.generate_prompt(res).split('.')[0]}.")
            # print(f"asdasd ::: {res}")
            book.append(f"{res}")
            st.text_area("PERSONA2",value=f"Persona {bot1.name} : {res}")
            print('\n'.join(book[-1:]))
            init_conversation2(bot1,bot2,res)
        else:
            st.error("Process stopped")

    def init_conv(botnum:int,persona1:Bot,persona2:Bot,topic:str):
        init_conversation1(persona1,persona2,topic) if botnum==1 else init_conversation2(persona1,persona2,topic)


    def edit_book():
        with st.sidebar:
            st.subheader("Select the Line of Book to Edit")
            selected=option_menu(
                    menu_title=None,  # required
                    options=[el[5:] for el in st.session_state.book],  # required
                    default_index=-1,  # optional
                )
            line_num=st.session_state.book.index(selected)
            newcontent=st.text_input("Enter New Line",value=st.session_state.book[line_num])
            if newcontent: 
                st.session_state.book=st.session_state.book[:line_num-1]+newcontent+st.session_state.book[line_num+1:]
                st.success(f'Book edited at line {line_num}')
            else:
                st.session_state.book=st.session_state.book[:line_num-1]+book[line_num+1:]
                st.success(f'Book line removed at Line {line_num}')

    def createpersona(**kwargs):
        newbot=Bot(**kwargs)
        st.session_state.all_personas.append(newbot)


    def create_persona_sidebar():
        with st.sidebar:
            st.subheader("Create Your Persona")
            persona1_name = st.sidebar.text_input("Name:", value="Tom")
            persona1_characteristics = st.sidebar.text_input("Characteristics:", "It is usually angry")
            persona1_interests = st.sidebar.text_input("Interests:", "It likes to fight alot")
            
            uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type=["txt"], key='text1')
            if uploaded_file is not None:
                try:
                    data=uploaded_file.read().decode('utf-8')
                    with open(f'file{persona1_name}.txt','w') as f:
                        f.write(data)
                except FileNotFoundError:
                    st.error("File not found. Please upload a valid text file.")
            
            if st.button("Create Persona"):
                createpersona(name=persona1_name,
                                characteristics=persona1_characteristics,
                                interests=persona1_interests,
                                knowledge_file=f'file{persona1_name}.txt')       
                print(len(st.session_state.all_personas))
                st.success("New Persona Created")
        return

    def edit_persona_sidebar():
        with st.sidebar:
            st.subheader("Select Persona to Edit")
            selected=option_menu(
                    menu_title=None,  # required
                    options=[el.name for el in st.session_state.all_personas],  # required
                    default_index=-1,  # optional
                )
            if selected:
                to_edit=[el.name for el in st.session_state.all_personas].index(selected)
                with st.sidebar:
                    st.subheader("Edit Your Persona")
                
                    selectedp=option_menu(
                            menu_title=None,  # required
                            options=["name","characteristics","interest"],  # required
                            default_index=0,  # optional
                        )        
                    if selectedp:
                        if selectedp=="name":
                            st.session_state.all_personas[to_edit].name=st.text_input(f'Change {selectedp}',value=f"{st.session_state.all_personas[to_edit].name}")    
                        if selectedp=="characteristics":
                            st.session_state.all_personas[to_edit].characteristics=st.text_input(f'Change {selectedp}',value=f"{st.session_state.all_personas[to_edit].characteristics}")
                        if selectedp=="interest":
                            st.session_state.all_personas[to_edit].interests=st.text_input(f'Change {selectedp}',value=f"{st.session_state.all_personas[to_edit].interests}")
                    if st.button("Make Changes"):
                        st.success("All Changes Made")
        return

    def initdialogue_sidebar():
        with st.sidebar:
            st.subheader("Select First Persona for Dialogue")
            selectp1=option_menu(
                    menu_title=None,  # required
                    options=[el.name for el in st.session_state.all_personas],  # required
                    default_index=-1,  # optional
                )
            st.subheader("Select Second Persona for Dialogue")
            selectp2=option_menu(
                    menu_title=None,  # required
                    options=[el.name for el in st.session_state.all_personas],  # required
                    default_index=0,  # optional
                )
        input_text = st.text_input("Enter Book Line:", placeholder="Write story starting e.g.a person goes out for fishing.", key="input")
        input_botnum = st.number_input("Enter a number:", min_value=1, max_value=2, step=1)
        b1=[el.name for el in st.session_state.all_personas].index(selectp1)
        b2=[el.name for el in st.session_state.all_personas].index(selectp1)
        init_conv(persona1=st.session_state.all_personas[b1],persona2=st.session_state.all_personas[b2],topic=input_text,botnum=input_botnum)

    def downloadbook():
        with open("downloaded_file.txt", "w") as file:
            file.write(''.join([line for line in st.session_state.book]))
        st.download_button(
                label="Download Book",
                data="BOOK.txt",
                key="BOOK",
                help="Click to download your Book",
            )
        st.session_state.book=[]

    def startapp():
        st.title("Persona Input App")

        selected = option_menu(
                menu_title=None,  # required
                options=["CreatePersona", "EditPersona", "InitiateDialogue","EditYourBook","Publishbook"],  # required
                default_index=0,  # optional
                orientation="horizontal",
            )
        if selected=="CreatePersona":
            create_persona_sidebar()
        
        if selected=="EditPersona":
            print(len(st.session_state.all_personas))
            if len(st.session_state.all_personas)>0:
                edit_persona_sidebar() 
            else: 
                st.error('No Personas created')
        
        if selected=="InitiateDialogue":
                if len(st.session_state.all_personas)>1: 
                    initdialogue_sidebar()
                else: 
                    st.error('Need Atleast 2 Personas to be created')

        if selected=="EditYourBook":
            st.session_state.book=['qwe','qwewqeqwe','ertertert']
            if len(st.session_state.book)>0:
                edit_book()
            else:
                st.error('No Book to Show')

        if selected=="Publishbook":
            if len(st.session_state.book)>0:
                downloadbook()
            else:
                st.error('No Book to Download')
                
                
    thread = threading.Thread(target=startapp())
    thread.start()
    stop_button = st.button("Stop Execution")
    if stop_button:
        stop_execution=False

        




if __name__=="__main__":
    main_page()
# persona1=createpersona(name='A',characteristics='it is usually angry',interests='it likes to watch action movies',knowledge_file='knowledge1.txt')
    # persona2=createpersona(name='B',characteristics='it is usually happy',interests='it likes to watch underage kids cartoons',knowledge_file='knowledge2.txt')
    
    # # print(persona1.knowledge(kpath="knowledge1.txt",query='person'))
    
    # topic='Ben goes outside with his girlfriend'
    # # while(True):
    # #     init_conversation1(persona1,persona2,'Ben goes outside with his girlfriend')
    
    # init_conv(1,persona1,persona2,topic)
    
    
    


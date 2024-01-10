

import os

import streamlit as st
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.qa_with_sources.retrieval import \
    RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
# from
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from . import treeofthoughts, utils


class BasePlaygroundBot():
    """
    A base class representing a playground bot.

    Attributes:
    -----------
    model_name : str
        The name of the model to use. Default is "gpt-4".
    llm : ChatOpenAI
        An instance of the ChatOpenAI class.
    description : str
        A description of the playground bot.

    Methods:
    --------
    ask(question: str) -> str:
        Asks the bot a question or gives it a prompt and returns the bot's response.
    getDescription() -> str:
        Returns the description of the playground bot.
    display(elem, result):
        Displays the bot's response in the specified element.
    """
    def __init__(self,model_name="gpt-4") -> None:
        """
        Initializes a new instance of the BasePlaygroundBot class.

        Parameters:
        -----------
        model_name : str
            The name of the model to use. Default is "gpt-4".
        """
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.description = "Blank Description"
    def ask(self,question:str)->dict:
        """
        Asks the bot a question or gives it a prompt and returns the bot's response.

        Parameters:
        -----------
        question : str
            The prompt or question to ask the bot.

        Returns:
        --------
        str
            The bot's response to the prompt or question.
        """
        pass
    def getDescription(self)->str:
        """
        Returns the description of the playground bot.

        Returns:
        --------
        str
            The description of the playground bot.
        """
        return self.description
    def display(self,elem,result):
        """
        Displays the bot's response in the specified element.

        Parameters:
        -----------
        elem : str
            The element to display the bot's response in.
        result : dict
            A dictionary containing the bot's response.
        """
        elem.write("empty bot")

class PlayGroundGPT4(BasePlaygroundBot):
    """
    A class representing a playground bot that uses the GPT-4 model.

    Attributes:
    -----------
    model_name : str
        The name of the model to use. Default is "gpt-4".
    chain : ConversationChain
        An instance of the ConversationChain class.
    description : str
        A description of the GPT-4 model.

    Methods:
    --------
    ask(prompt: str) -> str:
        Asks the bot a question or gives it a prompt and returns the bot's response.
    display(elem, result):
        Displays the bot's response in the specified element.
    """
    def __init__(self, model_name="gpt-4") -> None:
        """
        Initializes a new instance of the PlayGroundGPT4 class.

        Parameters:
        -----------
        model_name : str
            The name of the model to use. Default is "gpt-4".
        """
        super().__init__(model_name=model_name)
        self.chain = ConversationChain(llm=self.llm)
        self.description = "GPT4 is the latest version of GPT3. It is trained on a larger dataset and has more parameters. It is the most powerful language model in the world."
    
    def ask(self, prompt: str) -> str:
        """
        Asks the bot a question or gives it a prompt and returns the bot's response.

        Parameters:
        -----------
        prompt : str
            The prompt or question to ask the bot.

        Returns:
        --------
        str
            The bot's response to the prompt or question.
        """
        return self.chain(prompt)
    
    def display(self, elem, result):
        """
        Displays the bot's response in the specified element.

        Parameters:
        -----------
        elem : str
            The element to display the bot's response in.
        result : dict
            A dictionary containing the bot's response.
        """
        elem.write(result["response"])


class PlayGroundGPT4ToT(BasePlaygroundBot):
    """
    A class representing a playground bot that uses the Tree of Thought model.

    Attributes:
    -----------
    model_name : str
        The name of the model to use. Default is "gpt-4".
    chain : ConversationChain
        An instance of the ConversationChain class.
    description : str
        A description of the Tree of Thought model.

    Methods:
    --------
    ask(prompt: str) -> str:
        Asks the bot a question or gives it a prompt and returns the bot's response.
    display(elem, result):
        Displays the bot's response in the specified element.
    """
    def __init__(self, model_name="gpt-4") -> None:
        """
        Initializes a new instance of the PlayGroundGPT4ToT class.

        Parameters:
        -----------
        model_name : str
            The name of the model to use. Default is "gpt-4".
        """
        super().__init__(model_name=model_name)
        self.chain = ConversationChain(llm=self.llm)
        self.description = "The Tree of Thought is a conversational AI model developed by Langchain that uses GPT-4 as its underlying language model. It is designed to generate human-like responses to user input and can be used for a variety of applications, including chatbots, virtual assistants, and customer service."
    def ask(self, prompt: str) -> str:
        """
        Asks the bot a question or gives it a prompt and returns the bot's response.

        Parameters:
        -----------
        prompt : str
            The prompt or question to ask the bot.

        Returns:
        --------
        str
            The bot's response to the prompt or question.
        """
        return {"response":treeofthoughts.ask(prompt)}

    def display(self,elem,result):
        """
        Displays the bot's response in the specified element.

        Parameters:
        -----------
        elem : str
            The element to display the bot's response in.
        result : dict
            A dictionary containing the bot's response.
        """
        elem.write(result["response"])

class PlayGroundGPT4CoT(BasePlaygroundBot):
    """
    A class representing a playground bot that uses the CoT model.

    Attributes:
    -----------
    model_name : str
        The name of the model to use. Default is "gpt-4".
    chain : ConversationChain
        An instance of the ConversationChain class.
    description : str
        A description of the CoT model.

    Methods:
    --------
    ask(prompt: str) -> str:
        Asks the bot a question or gives it a prompt and returns the bot's response.
    display(elem, result):
        Displays the bot's response in the specified element.
    """
    def __init__(self, model_name="gpt-4") -> None:
        """
        Initializes a new instance of the PlayGroundGPT4CoT class.

        Parameters:
        -----------
        model_name : str
            The name of the model to use. Default is "gpt-4".
        """
        super().__init__(model_name=model_name)
        
        self.planllm = self.llm
        plan_prompt = PromptTemplate(
            template= """
        Come up with a plan to solve the following problem as if you were an experienced doctor.
        Problem:
        {problem}

        Come up with plan to research to solve the problem in steps:
        """, 
            input_variables=["problem"]
            )
        
        execution_prompt = PromptTemplate(
            template="""
            from this plan, tell the patient what they need to.
            {plan}
            Helpful Answer for a concerned clinic visitor :
            """,
            input_variables=["plan"]
            )
        
        self.chainPlan = plan_prompt | self.llm | StrOutputParser()
        self.chainResponse = execution_prompt | self.llm | StrOutputParser()

        self.description = "CoT prompting, as introduced in a recent paper, is a method that encourages LLMs to explain their reasoning process."
    def ask(self, prompt: str) -> str:
        """
        Asks the bot a question or gives it a prompt and returns the bot's response.

        Parameters:
        -----------
        prompt : str
            The prompt or question to ask the bot.

        Returns:
        --------
        str
            The bot's response to the prompt or question.
        """
        # this st.write works because it was called under st.status()
        st.write("creating plan")
        plan = self.chainPlan.invoke({"problem":prompt})
        st.write("the plan")
        st.caption(plan)
        st.write("getting solution from the plan")
        response = self.chainResponse.invoke({"plan":plan})
        return {
            "response":response,
            "plan":plan,
            }
    def display(self,elem,result):
        """
        Displays the bot's response in the specified element.

        Parameters:
        -----------
        elem : str
            The element to display the bot's response in.
        result : dict
            A dictionary containing the bot's response.
        """
        with elem:
            with st.expander("Plan"):
                st.write(result["plan"])
            
            st.write(result["response"])

class PlayGroundGPT4CoTChroma(BasePlaygroundBot):
    """
    A class representing a playground bot that uses the CoTChroma model.

    Attributes:
    -----------
    model_name : str
        The name of the model to use. Default is "gpt-4".
    chain : ConversationChain
        An instance of the ConversationChain class.
    description : str
        A description of the CoTChroma model.

    Methods:
    --------
    ask(prompt: str) -> str:
        Asks the bot a question or gives it a prompt and returns the bot's response.
    display(elem, result):
        Displays the bot's response in the specified element.
    """
    def __init__(self, model_name="gpt-4",path: str = "./.datalake/HC_DATA/prepared_generated_data_for_nhs_uk_conversations.csv") -> None:
        """
        Initializes a new instance of the PlayGroundGPT4CoTChroma class.

        Parameters:
        -----------
        model_name : str
            The name of the model to use. Default is "gpt-4".
        """
        super().__init__(model_name=model_name)
        self.chain = ConversationChain(llm=self.llm)
        self.description = "At its core, CoT prompting is about guiding the LLM to think step by step. This is achieved by providing the model with a few-shot exemplar that outlines the reasoning process. The model is then expected to follow a similar chain of thought when answering the prompt. \n Added vector database retrival of the source"
        self.template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible. 
        Always gives the answer in your own words, do not copy and paste from the context.
        Always give the reference to the source of the answer as links found from the context.
        response in markdown format
        HISTORY:
        {chat_history}
        QUESTION: 
        {question}
        Helpful Answer for a concerned clinic visitor :"""
        self.QA_CHAIN_PROMPT = PromptTemplate.from_template(self.template)
        self.llm = ChatOpenAI(temperature=0)

    
        if "memory" not in st.session_state: # if memory is not initialized
            st.session_state.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key='chat_history', return_messages=True, output_key='answer'
            )
            
        self.memory = st.session_state.memory

        if not os.path.exists("./.chroma_db"):
            loader = CSVLoader(file_path=path,csv_args={"quotechar": '"'})
            documents = loader.load_and_split()
            self.vectorstore = Chroma.from_documents(
                documents=documents, 
                embedding=OpenAIEmbeddings(),
                persist_directory="./.chroma_db",
                
                )
        else:
            self.vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./.chroma_db")

        
    def ask(self, prompt: str) -> dict:
        """
        Asks the bot a question or gives it a prompt and returns the bot's response.

        Parameters:
        -----------
        prompt : str
            The prompt or question to ask the bot.

        Returns:
        --------
        str
            The bot's response to the prompt or question.
        """
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),# ok
            retriever=self.vectorstore.as_retriever(), # ok
            condense_question_prompt = self.QA_CHAIN_PROMPT, # ok
            # chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT,"verbose":True},
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            )
        result = qa_chain({"question": prompt})
        result["response"] = result["answer"]
        return result
        

    def display(self,elem,result):
        """
        Displays the bot's response in the specified element.

        Parameters:
        -----------
        elem : str
            The element to display the bot's response in.
        result : dict
            A dictionary containing the bot's response.
        """

        with elem:
            st.write(result["answer"])
            with st.expander(f"Sources"):
                for i,source in enumerate(result["source_documents"]):
                    st.subheader(f"Sources {i}")
                    for chat in utils.split_document_chat(source.page_content):
                        role = chat["who"]
                        message = chat["message"]
                        elem.markdown(f"**{role.upper()}** {message}")





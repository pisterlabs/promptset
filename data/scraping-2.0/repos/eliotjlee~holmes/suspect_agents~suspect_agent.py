"""The SuspectAgent class encapsulates the behavior and memory of a suspect in a murder case.

Each SuspectAgent is capable of interacting with the user, providing responses to questions based on the suspect's
memory and the current plot of the story.

Methods:
    __init__(self, plot, i):
        Initializes the SuspectAgent with a reference to the plot and a suspect identifier.

    get_suspect_response(self, user_message):
        Processes a user message, retrieves relevant memories, and generates a suspect response.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


class SuspectAgent:
    def __init__(self, plot, i):
        """
        Initializes the SuspectAgent with a reference to the plot and a suspect identifier.

        Args:
        plot (object): The plot object containing the story and suspect information.
        i (int): The identifier of the suspect this agent represents.
        """
        self.suspect = plot.suspects[i]
        self.memory_path = self.suspect.memory_path
        self.chat_memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

        print(f"SUSPECT: {self.suspect.name}")
        print(f"MEMORY PATH: {self.memory_path}")

        # Load memory (suspect account)
        with open(self.memory_path) as f:
            memory_stream = f.read()

        embeddings = OpenAIEmbeddings()

        # Split text, upsert chunks into a ChromaDB instance
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        texts = text_splitter.split_text(memory_stream)

        # ChromaDB instance for similarity search + RAG
        self.docsearch = Chroma.from_texts(
            texts, embeddings
        )

        # Build prompt template for conversation chain
        template = "You are a suspect in a murder, and you are currently being questioned by the lead detective.\n\n"
        template += f"Here are the details about the case: \n\n{plot.get_this_suspect_summary(i)}\n\n"

        template += """

        Here are the memories that pop into your head as you hear the detective's question:

        {context}

        Selectively choose which to incorporate into your response. But NEVER quote them directly. These are your internal thoughts.
        """

        if self.suspect.guilty:
            template += """
            
            Remember, if you ARE guilty, you do not want the detective to know it. When the detective's question makes you think of the murder, you get very anxious and agitated and start to stutter.
            """
        else:
            template += """

            If the detective begins to wrongly accuse you of being guilty, you get really anxious and stutter a lot.
            """

        template += """
        
        {chat_history}

        Only answer this one question. Speak as your character would, based on their bio and tags--do not regurgitate these, but emobdy them. Do not be overly rigid and formal.

        Detective's question: {human_input}
        Your response to this question:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"], template=template
        )

        # Build chain to use as conversational agent
        self.chain = load_qa_chain(
            ChatOpenAI(
                temperature=0,
                model="gpt-4"
            ),
            chain_type="stuff",
            memory=self.chat_memory,
            prompt=prompt
        )

    def get_suspect_response(self, user_message):
        """
        Processes a user message, retrieves relevant memories from the suspect's memory,
        and generates a response based on the suspect's characteristics and the current context.

        Args:
        user_message (str): The message from the user to which the suspect should respond.

        Returns:
        str: The suspect's response to the user's message.
        """
        doc_search = self.docsearch.similarity_search(user_message, 4)
        response = self.chain({"input_documents": doc_search, "human_input": user_message}, return_only_outputs=True)
        return response.get("output_text")

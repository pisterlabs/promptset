from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.summary import SummarizerMixin
from langchain.prompts.prompt import PromptTemplate

from embed import Embedder


def _print_citations(source_docs):
    print('References : ')
    for sd in source_docs:
        if sd.metadata.get('source', ''):
            file = sd.metadata.get('source')
            data, course, lecture_file = file.split('/')
            lecture, filetype = lecture_file.split('.')
            page = sd.metadata.get('page', '')
            if filetype == 'pdf':
                print('Course : ', course.upper(), ' Lecture : ', lecture.split('.')[0], ' Page : ', page)
            else:
                print('Course : ', course.upper(), ' Lecture : ', lecture.split('.')[0])


def _start_chat():
    print("Instructions : Enter your queries in stdin when prompted. The agent will reply.")
    print("To end the conversation, enter 'exit'. The agent will print a summary of the conversation")
    print("Starting the Conversation Agent")
    print("...")


class Conversation:
    def __init__(self, debug_mode=False, courses=None):
        if courses is None:
            self.courses = ['cs231n', 'cs234', 'cs324']
        else:
            self.courses = courses
        if len(self.courses) == 3:
            self.embedder = Embedder()
        if len(self.courses) == 1:
            self.embedder = Embedder(persist_directory='chroma_databases/db_'+self.courses[0])
        self.vectorstore = self.embedder.get_vectorstore()
        self.llm = ChatOpenAI(
            temperature=0.0,
            model_name="gpt-3.5-turbo")

        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            output_key='answer',
            memory_key='chat_history',
            return_messages=True)

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"include_metadata": True})

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            get_chat_history=lambda h: h,
            verbose=debug_mode)
        self.debug = debug_mode
        self.hot_start()

    def hot_start(self):
        if 'cs231n' in self.courses:
            self.chain({'question': "What are convolutional neural networks and how are they used ?"})
        if 'cs324' in self.courses:
            self.chain({'question': "In large language models, what are some milestone model architectures and papers "
                                    "in the last few years?"})
        if 'cs234' in self.courses:
            self.chain({'question': "What is reinforcement learning and what are its applications?"})

    def achat(self):
        query = input("Query : ")
        if query == 'exit' or '':
            return False
        ret = self.chain({'question': query})
        if self.debug:
            print(ret)
        print('Answer : ', ret.get('answer'))
        source_docs = ret.get('source_documents')
        _print_citations(source_docs)
        return True

    def chat(self):
        _start_chat()
        continue_conversation = self.achat()
        while continue_conversation:
            continue_conversation = self.achat()
        self.summarize_conversation()

    def summarize_conversation(self):
        SUMMARIZER_TEMPLATE = """Progressively summarize the lines of conversation provided, adding onto the previous 
        summary returning a new summary. Format the summary into bullet points. Keep the summary very short. EXAMPLE 
        Current summary: The human asks what the AI thinks of artificial intelligence. The AI thinks artificial 
        intelligence is a force for good.

            New lines of conversation:
            Human: Why do you think artificial intelligence is a force for good?
            AI: Because artificial intelligence will help humans reach their full potential.

            New summary:
             - Artificial intelligence is a force for good
             - It will help humans reach their full potential.
            END OF EXAMPLE

            Current summary:
            {summary}

            New lines of conversation:
            {new_lines}

            New summary:"""
        FINAL_SUMMARY_PROMPT = PromptTemplate(
            input_variables=["summary", "new_lines"], template=SUMMARIZER_TEMPLATE
        )
        sm = SummarizerMixin(llm=self.llm, prompt=FINAL_SUMMARY_PROMPT)
        print(sm.predict_new_summary(self.memory.chat_memory.messages, ""))


if __name__ == "__main__":
    from configs import set_keys

    set_keys()
    c = Conversation(debug_mode=True, courses=None)
    c.chat()

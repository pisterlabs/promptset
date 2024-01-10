from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from rich.panel import Panel

from cauldron import Cauldron
from cli import console
from utils import Agent


class Witch:
    def __init__(self, client, directory: str, history: dict) -> None:
        self.client = client
        self.directory = directory
        self.cauldron = None  # only initialize cauldron after witch is called

        self.agent = Agent.WITCH
        history[self.agent] = []
        self.history = history[self.agent]

    def __build_qa(self) -> None:
        self.cauldron = Cauldron(self.directory)

        llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True
        )
        cauldron = self.cauldron.get_db()
        retriever = cauldron.as_retriever(lambda_val=0.025, k=5, filter=None)
        template = (
            'The following is a conversation between a human and an AI '
            + "assistant. The AI answers the human's questions ONLY from the "
            + "human's notes while providing lots of details. If the AI does "
            + 'not know the answer to the question, it truthfully says that it '
            + 'does not know.\n'
            + 'History of conversation:\n'
            + '{context}\n'
            + '\n'
            + 'Conversation:\n'
            + 'Human: {question}\n'
            + 'AI:'
        )
        prompt = PromptTemplate(
            input_variables=['context', 'question'], template=template
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={'prompt': prompt},
        )

    def answer_question(self, question: str) -> None:
        if self.cauldron is None:
            self.__build_qa()

        res = self.qa({'question': question})
        answer = res['answer']
        console.print(Panel(answer, title=self.agent.value))
        self.history.append((question, answer))

    def reingest(self) -> None:
        self.cauldron.reingest()

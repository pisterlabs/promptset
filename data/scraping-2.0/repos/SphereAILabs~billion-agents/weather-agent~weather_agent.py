from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    Document,
)
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from prompts import SYSTEM_MESSAGE


class WeatherAgent:
    """
    An ReACT agent that answers questions about the weather by looking results up on Google Search. This is
    a "Hello World" example of an agent. Almost all queries should take 2 steps to complete.

    The flow goes as follows:
    1) Search google for answer
    2) Answer question

    The agent has no short term or long term memory. Every time you ask the agent a question, it does not
    have any memory of previous questions.

    When the agent uses Google Search, it only looks at the first 2 documents based on the heuristic
    that the top most result is probably the most relevant. Text from Google Search result is unstructured
    so weather results may or may not be accurate as it depends on how GPT decides to parse it.
    """

    def __init__(self, temperature=0.0, verbose=True, max_iterations=2):
        self.temperature = temperature
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.model = "gpt-3.5-turbo"
        self.stop = ["\nObservation:"]
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
        )
        self.system_prompt = SystemMessage(content=SYSTEM_MESSAGE)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        self.embeddings = HuggingFaceEmbeddings()

    def _create_question(self, question: str) -> HumanMessage:
        return HumanMessage(content=f"Question: {question}")

    def _create_observation(self, observation: str) -> HumanMessage:
        return HumanMessage(content=f"Observation: {observation}")

    def _search(self, query: str) -> Document:
        # search google
        google_search_doc = self._google_search(query)

        # split the search doc
        docs = self.splitter.split_documents([google_search_doc])

        # get the top 3 results
        top_result = docs[0:3]

        content = ""
        for doc in top_result:
            content += doc.page_content

        doc = Document(page_content=content, metadata=top_result[0].metadata)

        return doc

    def _google_search(self, query: str) -> Document:
        uri = f"https://google.com/search?q={query}"
        loader = WebBaseLoader(uri)
        docs = loader.load()
        doc = docs[0]

        return doc

    def _is_final_answer(self, message: BaseMessage):
        content = message.content
        return "Final Answer: " in content

    def _get_final_answer(self, message: BaseMessage):
        content = message.content.strip()
        final_anwser_prefix = "Final Answer: "
        index = content.index(final_anwser_prefix)
        final_answer = content[index + len(final_anwser_prefix) :]

        return final_answer

    def _get_action_and_input(self, message: BaseMessage) -> dict:
        content = message.content.strip()
        action_prefix = "Action: "
        action_index = content.index(action_prefix)

        unparsed_action = content[action_index + len(action_prefix) :]
        action, action_input = unparsed_action.split("[")
        action_input = action_input[:-1]

        return {"action": action, "action_input": action_input}

    def query(self, question: str) -> str:
        """
        Ask the agent a question about the weather.
        """
        question_message = self._create_question(question)
        messages = [self.system_prompt, question_message]

        iteration = 0

        def should_loop():
            if self.max_iterations == None:
                return True

            return iteration < self.max_iterations

        # agent runs loop until it gives a Final Answer
        while should_loop():
            iteration += 1
            message = self.llm(messages, stop=self.stop)

            # add ai message to chat messages
            messages.append(message)

            if self.verbose:
                print(message.content)
                print("-----")

            # check if message contains Final Answer: -> stop the loop
            contains_final_answer = self._is_final_answer(message)

            # get the final answer if it exists
            if contains_final_answer:
                final_answer = self._get_final_answer(message)
                return final_answer

            # parse action the agent wants to perform
            action_dict = self._get_action_and_input(message)
            action, action_input = action_dict["action"], action_dict["action_input"]

            if action == "Search":
                # search query on Google
                search_result = self._search(action_input)

                # feed observation to the agent
                observation = self._create_observation(search_result.page_content)
                print(observation.content)
                messages.append(observation)

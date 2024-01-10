# from ..lgbert.bert_pytorch.predict_log import Predictor
from langchain.chat_models import ChatOpenAI as OpenAI
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool

load_dotenv()


class Detection:
    def __init__(self, options):
        # self.model = Predictor(options=options)
        self.prompt = "Act as monitoing tool for deployed systems and detect anamolies in the logs. Answer with error line and type of anomaly as python tuple"
        self.temperature = 0.4
        self.max_tokens = 2048
        self.model_engine = "gpt-3.5-turbo"
        self.llm = OpenAI(model_name=self.model_engine,
                          temperature=self.temperature, max_tokens=self.max_tokens)

    def _search(query):
        with DDGS() as ddgs:
            for r in ddgs.text(query):
                return r

    def get_anamoly(self, logs) -> bool:
        tools = [
            Tool(
                name="search",
                func=self._search,
                description="Useful when you need to search for something on the internet.",
            ),]

        # Generate a solution using OpenAI's GPT-3 API
        prompt = f"{self.prompt}\n\nLogs:\n{logs}\n\nSolution:"

        agent = initialize_agent(
            tools, self.llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)
        answer = agent.run(prompt)
        return answer


if __name__ == "__main__":
    detect = Detection(None)
    print(detect.get_anamoly('''2023-11-05 16:44:27 
2023-11-05 16:44:27 Hello from Docker!
2023-11-05 16:44:27 This message shows that your installation appears to be working correctly.
2023-11-05 16:44:27 
2023-11-05 16:44:27 To generate this message, Docker took the following steps:
2023-11-05 16:44:27  1. The Docker client contacted the Docker daemon.
2023-11-05 16:44:27  2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
2023-11-05 16:44:27     (arm64v8)
2023-11-05 16:44:27  3. The Docker daemon created a new container from that image which runs the
2023-11-05 16:44:27     executable that produces the output you are currently reading.
2023-11-05 16:44:27  4. The Docker daemon streamed that output to the Docker client, which sent it
2023-11-05 16:44:27     to your terminal.
2023-11-05 16:44:27 
2023-11-05 16:44:27 To try something more ambitious, you can run an Ubuntu container with:
2023-11-05 16:44:27  $ docker run -it ubuntu bash
2023-11-05 16:44:27 
2023-11-05 16:44:27 Share images, automate workflows, and more with a free Docker ID:
2023-11-05 16:44:27  https://hub.docker.com/
2023-11-05 16:44:27 
2023-11-05 16:44:27 For more examples and ideas, visit:
2023-11-05 16:44:27  https://docs.docker.com/get-started/
2023-11-05 16:44:27 '''))

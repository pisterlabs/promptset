from langchain.prompts import load_prompt
from langchain.chat_models import GigaChat
from langchain.chains import LLMChain
from langchain.schema import SystemMessage

class NewsAgent:
    def __init__(self, auditory_name: str):
        self.llm = GigaChat(
            verbose=True,
            temperature=1,
            model="GigaChat:latest",
            credentials="MjhlMGRkYzYtNTdmNC00N2FmLTk5MmMtNmI0N2EzODJhOTM3OjgzYzZmZDQ3LWU5N2MtNDA3Ni1hOGI4LThhN2Q4ZmNiZWU2Mw==",
            #base_url="https://gigachat.devices.sberbank.ru/api/v1",
            #verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            timeout=300,
        )
        self.auditory_name = auditory_name

    def Run(self, news: str) -> str:
        response = self.first_step(news=news)

        if response.get("ответ") == "ПОДХОДИТ":
            pretty_news = self.second_step(news=news)
            print(pretty_news)
            return pretty_news
        else:
            print(response)
            return None


    def first_step(self, news: str) -> dict[str, str]:
        try:
            check_news_prompt = load_prompt('prompts/check_news.yaml')
            text = check_news_prompt.format(auditory_name=self.auditory_name) + '\n' + news

            raw_response = self.llm([SystemMessage(content=text)]).content
            answer = raw_response.split("\n")[0]
            explanation = raw_response.split("\n")[1]

            response = {
                "ответ": f"{answer}",
                "обоснование": f"{explanation}"
            }

            return response

        except Exception as e:
            print(f"Unable to generate chat response: {e}")

    def second_step(self, news: str) -> str:
        try:
            pretty_news_output_prompt = load_prompt('prompts/pretty_news_output.yaml')
            text = pretty_news_output_prompt.format(auditory_name=self.auditory_name) + '\n' + news

            raw_response = self.llm([SystemMessage(content=text)]).content
            return raw_response
        
        except Exception as e:
            print(f"Unable to generate chat response: {e}")

import requests
from bs4 import BeautifulSoup
from langchain.tools import BaseTool


class InternetTool(BaseTool):
    name: str = "internet_tool"
    description: str = (
        "useful when you want to read the text on any url on the internet."
    )

    def get_text_content(self, url: str) -> str:
        """Get the text content of a webpage with HTML tags removed"""
        response = requests.get(url)
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")
        for tag in ["nav", "footer", "aside", "script", "style", "img"]:
            for match in soup.find_all(tag):
                match.decompose()

        text_content = soup.get_text()
        text_content = " ".join(text_content.split())
        return text_content

    def limit_chars(self, text: str) -> str:
        """limit number of output characters"""
        return text[:10_000]

    def _run(self, url: str) -> str:
        try:
            text_content = self.get_text_content(url)
            return self.limit_chars(text_content)
        except Exception as e:
            return f"The following error occurred while trying to fetch the {url}: {e}"

    def _arun(self, url: str):
        raise NotImplementedError("This tool does not support asynchronous execution")


if __name__ == "__main__":
    tool = InternetTool()
    print(
        tool.run("https://en.wikipedia.org/wiki/List_of_Italian_desserts_and_pastries")
    )

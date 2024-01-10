from typing import Optional
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


class YesNoOutputParser(Runnable):
    def invoke(self, result:str, config: Optional[RunnableConfig] = None) -> bool:
        result_char = result.strip()[:1].lower()
        if result_char == "y":
            return True
        elif result_char == "n":
            return False
        
        raise ValueError(f"Invalid result: {result}")


class RagParser(Runnable):
    def invoke(self, result, config: Optional[RunnableConfig] = None) -> bool:
        text = result['answer']
        text += "\n----------------------\n"
        for i, document in enumerate(result['documents']):
            pmid = document['pmid']
            title = document['title']
            date = document['date']
            text += f"[{i+1}] {title} ({date}) - https://pubmed.ncbi.nlm.nih.gov/{pmid}\n\n"

        return text
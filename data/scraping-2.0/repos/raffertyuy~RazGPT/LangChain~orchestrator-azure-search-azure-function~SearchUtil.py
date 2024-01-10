from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

from azure.search.documents import SearchClient
import Prompts


def RephraseQuery(question: str, chat_history: dict, chatLLM: AzureChatOpenAI):
    searchprompt = Prompts.SEARCH_SYSTEMPROMPT.format(
        chat_history=chat_history,
        question=question)
    
    searchquery = chatLLM([HumanMessage(content=searchprompt)]).content
    return searchquery


def Search(query: str, top: int, client: SearchClient):
    results = client.search(query, top=top)

    resultslist = []
    sourcelist = []
    for result in results:
        resultslist.append(result['content'])
        sourcepage = result['sourcepage']
        sourcelist.append(
            {"sourcefile": result['sourcefile'], "sourcepage": abs(int(sourcepage[sourcepage.rfind('-'):sourcepage.rfind('.')]))})

    # Convert the list to a string
    resultscontent = "\n".join(resultslist)
    return resultscontent, sourcelist

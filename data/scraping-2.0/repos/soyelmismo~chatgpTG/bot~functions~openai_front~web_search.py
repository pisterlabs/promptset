from bot.src.utils.constants import ERRFUNC, FUNCNOARG
from bot.src.utils.gen_utils.openai.openai_functions_extraction import openaifunc
from bot.src.apis import duckduckgo
@openaifunc
async def search_on_internet(self, query: str, search_type: str, timelimit: str = None) -> str:
    """
    Search information/recommendations and news on internet
    Reveives a search query to search information and recommendations on the web. talk freely about the results.

    Args:
        query (str): the text that will be searched
        search_type (str): use "text" or "news" depending of what the user has requested
        timelimit (str): use "d" if latest results from today, for other time limits: "w", "m", "y". Defaults to None. they are d(day), w(week), m(month), y(year).
    
    Returns:
        str: the search / news results to inform the user
    """
    if query:
        try:
            return await duckduckgo.search(self, query = query, gptcall = True, timelimit = timelimit, type = search_type)
        except Exception: return ERRFUNC
    else: return FUNCNOARG
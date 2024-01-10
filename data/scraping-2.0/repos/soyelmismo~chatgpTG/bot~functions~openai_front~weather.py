from bot.src.utils.constants import ERRFUNC, FUNCNOARG
from bot.src.utils.gen_utils.openai.openai_functions_extraction import openaifunc
from bot.src.apis import wttr
@openaifunc
async def lookup_weather(self, location: str, unit: str) -> str:
    """
    Search actual weather info.

    Args:
        location (str): the city. mandatory.
        unit: "C" or "F". mandatory, and depends of the city

    Returns:
        str: all the weather info to be tell to the user
    """
    if location:
        try:
            return await wttr.getweather(location = location, unit = unit)
        except Exception: return ERRFUNC
    else: return FUNCNOARG

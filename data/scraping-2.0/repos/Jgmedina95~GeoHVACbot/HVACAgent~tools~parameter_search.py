from typing import Any, Coroutine
from langchain.tools import BaseTool
import pandas as pd

ground_temperatures = pd.read_csv('GroundTemperatues.csv')

class ground_temperature_search_tool(BaseTool):

    name= "Design Parameter Search Tool"
    description = """Searches for ground temperature needed for the pipe design
    input: (keyword arguments) state/country city. Optionally, you can also add the continent to get an approximate value. 
    
    examples: 
    Input: State=New York, City=New York
    Output: Ground Temperature and recommended working temperatures
    
    Input: Country=Ecuador, City=Quito
    output: 60.0 F
    
    Input:"""

    def _run(self,query):
    #split from commas
        query = query.split(',')
        #split from equal sign
        queries = {q.split('=')[0].strip(): q.split('=')[1].strip() for q in query}
        #from the dataframe get CWT (F) for either the state or  city
        result = pd.DataFrame()  # Empty DataFrame to store results
    
         # Check for both state and city
        if 'state' in queries and 'City' in queries:
            result = ground_temperatures[(ground_temperatures['State'] == queries['State']) & 
                                     (ground_temperatures['City'] == queries['City'])]
    
        # If no result found for state and city pair, or only state is provided, fetch based on state
        if result.empty or 'City' not in queries:
            result = ground_temperatures[ground_temperatures['State'] == queries['State']]
    
        if result.empty:
            return "No ground temperature found for given location, try with a close city or state or just retry with the continent, and an estimated value will be given"
        Temp = result['GWT (F)'].mean()
        return f"{Temp} Fahrenheit for ground temperature. Recommended working temperatures are {Temp+20} to {Temp+30} for cooling, and {Temp-10} to {Temp-20} for heating purposes."
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
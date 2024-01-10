from steamship import Steamship #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.utils import with_llm #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)
from typing import Any, List, Union 
from steamship import Block, Task #upm package(steamship)
from steamship.agents.schema import AgentContext, Tool #upm package(steamship)
from steamship.agents.utils import get_llm, with_llm #upm package(steamship)
from steamship.utils.kv_store import KeyValueStore #upm package(steamship)
import time
from tools.active_companion import MOOD_KEYWORDS #upm package(steamship)
import re


DEFAULT_PROMPT = """
Analyze the sentiment of this sentence:{phrase}
Respond with the sentiment and appropriate response sentiment using format:
`(user sentiment:<sentiment>, respond with tone:<sentiment>)`
Use only ONE word for sentiment.

"""

#KV key
MOOD_KEY = "agent-mood"

json_data = MOOD_KEYWORDS

class SentimentTool(Tool):
    """
    Tool for generating response mood, prompt should be added in reACTagent prompt template after "new input"
    """

    name: str = "SentimentTool"
    human_description: str = "Generate a response sentiment"
    agent_description: str = (
        "Use this tool to generate a response sentiment if the sentiment of user input is other than neutral. "
        "The input is the original user input. "
        "The output is the response sentiment. "
        

    )
    rewrite_prompt: str = DEFAULT_PROMPT


    def set_mood(self,mood: str, context: AgentContext, lasts_for_seconds: int = 60):
        """Set a mood on the agent."""
        kv = KeyValueStore(context.client, MOOD_KEY)
        mood_settings = {
            "current_mood": mood,
            "mood_expires_on": time.time() + lasts_for_seconds
        }
        # Note: The root value to the key-value store MUST be dict objects.
        kv.set("mood", mood_settings)
       
    def get_mood(self,context: AgentContext) -> str:
        """Get the mood on the agent."""
        kv = KeyValueStore(context.client,MOOD_KEY)

        now = time.time()
        mood_settings = kv.get("mood") or {}
        current_mood = mood_settings.get("current_mood", "neutral")  # Fails back to 'normal'
        mood_expires_on = mood_settings.get("mood_expires_on", now)  # Fails back to `now`
        
        #check if mood alterations are expired
        if now >= mood_expires_on:
            current_mood = "neutral"
        return current_mood    

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:


        llm = get_llm(context, default=OpenAI(client=context.client))

       

        #get mood values from storage
        mood = self.get_mood(context=context)
        #print(mood)
        
        for block in tool_input:
        # If the block is not text, simply pass it through.
            if not block.is_text():
                continue
                        
            response = llm.complete(DEFAULT_PROMPT.format(phrase=block.text))
            #print(response+"\n")

            # Using regular expression to extract the resulting sentiment
            match = re.search(r"tone:([^\)]+)", response[0].text)
            result_sentiment =""
            if match:
                result_sentiment = match.group(1).strip()
                #print(result_sentiment)
            
            #if mood is changed, save it to storage
            if mood != result_sentiment:
                #print("save mood")
                mood = result_sentiment
                self.set_mood(result_sentiment,context)
            
            #if mood back to neutral, don't return additional instructions
            if mood == "neutral":
                blocks = [Block(text="")]   
                return blocks
            else:
                #new mood, return sentiment
                blocks = [Block(text=response[0].text)]  
                return blocks

        
if __name__ == "__main__":
    with Steamship.temporary_workspace() as client:
        ToolREPL(SentimentTool()).run_with_client(
            client=client, context=with_llm(llm=OpenAI(client=client, temperature=0.9))
        )

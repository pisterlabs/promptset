from steamship import Steamship #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.utils import with_llm #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)
from typing import Any, List, Union #upm package(steamship)
from steamship import Block, Task #upm package(steamship)
from steamship.agents.schema import AgentContext, Tool #upm package(steamship)
from steamship.agents.utils import get_llm, with_llm #upm package(steamship)
from steamship.utils.kv_store import KeyValueStore #upm package(steamship)
import time
import json
from tools.active_companion import MOOD_KEYWORDS #upm package(steamship)
 
#additional tone instruction for input

DEFAULT_PROMPT = """You MUST Answer with a mood: {tone}"""

#KV key
MOOD_KEY = "agent-mood"

json_data = MOOD_KEYWORDS

class MoodTool(Tool):
    """
    Tool for generating response mood, prompt should be added in reACTagent prompt template after "new input"
    """

    name: str = "MoodTool"
    human_description: str = "Generate a response mood"
    agent_description: str = (
        "Use this tool to generate a response mood for the original question "
        "The input is the question or topic"
        "The output is the special response mood."
        

    )
    rewrite_prompt: str = DEFAULT_PROMPT


    def set_mood(self,mood: str, mood_level: int, keywords: str, context: AgentContext, lasts_for_seconds: int = 60):
        """Set a mood on the agent."""
        kv = KeyValueStore(context.client, MOOD_KEY)
        mood_settings = {
            "mood_level": mood_level,
            "current_mood": mood,
            "mood_expires_on": time.time() + lasts_for_seconds,
            "keywords": keywords
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
        keywords = mood_settings.get("keywords","")
        mood_level = mood_settings.get("mood_level",5)
        
        #check if mood alterations are expired
        if now >= mood_expires_on:
            current_mood = "neutral"
            mood_level = 5
            keywords = ""
        return current_mood, keywords,mood_level        

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:

        # Parse the JSON data
        data = json.loads(json_data)

        # Extract mood keywords and values
        mood_keywords = data["mood_keywords"]

        #get mood values from storage
        mood,keywords,mood_level = self.get_mood(context=context)
        #print(keywords)
        #print("current mood: "+mood)
        #print(mood_level)
        
        for block in tool_input:
        # If the block is not text, simply pass it through.
            if not block.is_text():
                continue
            #current mood
            mood_base = mood_level
            for keyword_data in mood_keywords:
                keyword = keyword_data["keyword"]
                value = keyword_data["value"]
                if keyword in block.text.lower() and keyword not in keywords.lower():
                    mood_level += value
                    keywords += keyword+","
                    #print("Keyword:", keyword)
                    #print("Value:", value)
                    #print("---")
               
            if mood_level != mood_base:           
                #check if new mood is applied
                if mood_level > 6:
                    mood = "enthusiastic"
                if mood_level > 5:
                    mood = "happy"
                if mood_level == 5:
                    mood = "neutral"
                if mood_level < 5:
                    mood = "sarcastic"
                if mood_level < 4:
                    mood = "grumpy"                
                
                #if mood is changed, save it to storage
                self.set_mood(mood,mood_level,keywords,context)

            #if mood back to normal, don't return additional instructions
            if "neutral" in mood and mood_level == mood_base:
                blocks = [Block(text="")]   
                return blocks
            else:
                #new mood, return mood instructions
                result = self.rewrite_prompt.format(tone=mood)
                blocks = [Block(text=result)]  
                return blocks

        
if __name__ == "__main__":
    with Steamship.temporary_workspace() as client:
        ToolREPL(MoodTool()).run_with_client(
            client=client, context=with_llm(llm=OpenAI(client=client, temperature=0.9))
        )

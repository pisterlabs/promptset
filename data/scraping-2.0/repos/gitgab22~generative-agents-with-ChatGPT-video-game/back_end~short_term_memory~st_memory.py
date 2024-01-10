import json
import openai
import agent
from back_end.short_term_memory import short_prompt as sp



# if long term memory is a dictionary attached to an agent
# In map evenement=[context,embedding]


class Short_Memory:
    lim = 0.3    # The limit value of importance

    def __init__(self, id):
        self.keep = []
        self.id = id

    
    def filter(self,agent,sensory_memory,time):
        events = "{"
        L = len(sensory_memory)
        for i in range(L):
            events += sensory_memory[i][0] + " ;\n"
        events += "}"
        ans = sp.get_completion(sp.p2_ger(agent, self.id, events, time))
        coef = json.loads(ans)
        for i in range(L):
            if coef[i] >= Short_Memory.lim:
                self.keep.append({sensory_memory[i][0]:coef[i]})
        
    def empty(self):
        self.keep = []

    
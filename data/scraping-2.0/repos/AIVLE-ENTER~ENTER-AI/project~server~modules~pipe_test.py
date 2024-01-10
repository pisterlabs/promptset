from langchain.memory import ConversationBufferMemory
import pickle
import json
# memory = ConversationBufferMemory(return_messages=True)
# memory.save_context({"input": "hi"}, {"output": "whats up"})
# memory.save_context({"input": "hello"}, {"output": "dark"})
# temp = memory.load_memory_variables({})
# memory2 = ConversationBufferMemory(return_messages=True)
# #temp['history'][2:]
# for i in range(len(temp['history'])//2):
#     memory2.save_context({"input": temp['history'][2*i].content},{"output": temp['history'][2*i+1].content})
#print(temp['history'][0].content)    
# print(memory.load_memory_variables({}))    
# print(memory2.load_memory_variables({}))
# print(memory==memory2)
#print(memory.json())
class ConversationBufferMemory_new(ConversationBufferMemory):
    def mem_load_k(self, k:int):
        temp = self.load_memory_variables({})['history']
        N_con = len(temp)//2
        if k >= N_con:
            return self
        else:
            memory2 = ConversationBufferMemory_new(return_messages=True, output_key="answer", input_key="question")
            for i in range(N_con-k,N_con):
                memory2.save_context({"question": temp[2*i].content},{"answer": temp[2*i+1].content})
            
            return memory2
    
    def conversation_json(self) -> str:
        temp = self.load_memory_variables({})['history']
        n=len(temp)//2
        d={'n': n, 'conversation':[]}
        for i in range(n):
            d['conversation'].append({'question':temp[2*i].content,'answer': temp[2*i+1].content})
        return j
                
def mem_load_k(memory:ConversationBufferMemory, k:int) -> ConversationBufferMemory:
    temp = memory.load_memory_variables({})['history']
    N_con = len(temp)//2
    if k >= N_con:
        return memory
    else:
        memory2 = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
        for i in range(N_con-k,N_con):
            memory2.save_context({"question": temp[2*i].content},{"answer": temp[2*i+1].content})
        
        return memory2

def conversation_json(memory:ConversationBufferMemory):
    temp = memory.load_memory_variables({})['history']
    n=len(temp)//2
    d={'n': n, 'conversation':[]}
    for i in range(n):
        d['conversation'].append({'question':temp[2*i].content,'answer': temp[2*i+1].content})
    #j = json.dumps(d,ensure_ascii=False, indent=3)
    return d

# history = 'cafe'
# with open('history/'+history+'.pkl','rb') as f:
#     memory = pickle.load(f)

# memory2 = mem_load_k(memory,3)


# print(conversation_json(memory))
# print('\n\n')
# print(memory2.load_memory_variables({}))
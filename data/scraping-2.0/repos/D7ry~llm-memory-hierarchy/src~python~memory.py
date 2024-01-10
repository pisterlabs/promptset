import os
import openai
#TODO: make cache limit customizable by user through agent
L1_CACHE_LIMIT: int = 1000
L1_CACHE_FLUSH_THRESHOLD: int = int(0.3 * L1_CACHE_LIMIT) # once L1 cache hits limit, flush L1 cache until it has less than L1_CACHE_FLUSH_THRESHOLD characters
class Context:
    def __init__(self, l1_memory: 'list[tuple[str, str]]', l2_memory: 'list[str]', l3_memory: 'list[str]', summary: str):
        self.most_recent_conversations: list[tuple[str, str]] = l1_memory
        self.l2_memory: list[str] = l2_memory
        self.l3_memory: list[str] = l3_memory
        self.summary: str = summary


        
class Memory:
    def __init__(self, memory_dict: 'dict'):
        self.__L1_cache: list = memory_dict["L1"]  # List to store recent conversations
        self.__L2_cache: list = memory_dict["L2"]  # List to store synopses of conversations
        self.__summary: str = memory_dict["summary"]


    def memorize(self, user_input: str, agent_input: str):
        """
        Memorize a conversation between the user and agent.
        Internally, cache user and agent input into the L1 cache.
        Automatically flush L1 cache to L2 when L1 cache is full, and flush L2 cache to L3 when L2 cache is full. 
        """
        self.__L1_cache.append(
                (user_input, agent_input)
                )
        self.__check_and_flush_cache()
       
    def serialize(self):
        return {
                "L1": self.__L1_cache,
                "L2": self.__L2_cache,
                "summary": self.__summary
                }

    def __check_and_flush_cache(self):
        """
        Checks if each level of the cache is full, and if is, flush to the next level.
        Currently we only use FIFO for eviction policy.
        """
        # check L1 cache
        l1_char_count: int = 0
        for conversation in self.__L1_cache:
            l1_char_count += len(conversation[0]) + len(conversation[1])
        print("current L1 cache size: ", l1_char_count) 
        if l1_char_count > L1_CACHE_LIMIT:
            print("L1 cache overflow detected! flushing to L2 cache...")
            # flushing L1 cache to L2 
            openai_client: openai.OpenAI = openai.OpenAI()
            # accumulate conversations to flush
            to_flush: list[tuple[str, str]] = []
            while l1_char_count > L1_CACHE_FLUSH_THRESHOLD:
                to_flush.append(self.__L1_cache.pop(0))
                l1_char_count -= len(to_flush[-1][0]) + len(to_flush[-1][1])
            
            messages: list = [
                        {
                        "role" : "system",
                        "content" : self.__summary
                        }
                    ]
            for conversation in to_flush:
                messages.append({"role" : "user", "content":conversation[0]})
                messages.append({"role" : "assistant", "content": conversation[1]})
            
            messages.append({"role": "user", 
                             "content" : "Please give a summary of all of the above conversations. The summary should be concise and captures the essense of the conversations. Important details should be retained, especially details regarding names. The summary should be made in first-person perspective from the assistant's perspective. Summary: "})
            completion = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages = messages
                    )
            summarized: str|None= completion.choices[0].message.content
            if summarized is not None: 
                self.__L2_cache.append(summarized)

    def __get_relevant_l3_cache(self, user_input: str) -> 'list[str]':
        #TODO: implement L3 cache fetching
        raise NotImplementedError("L3 cache not implemented yet!")

    def get_memory(self, user_input: str) -> Context:
        #l3_memory: list[str] = self.__get_relevant_l3_cache(user_input)
        l3_memory = []
        ret = Context(self.__L1_cache, self.__L2_cache, l3_memory, self.__summary)
        return ret

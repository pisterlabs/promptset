import json
import os
from datetime import datetime
from typing import Dict, List

from friend_replica.format_chat import format_chat_history, split_chat_data
from friend_replica.semantic_search import Chat
from langchain.prompts import PromptTemplate


class LanguageModelwithRecollection():
    '''
    Wrap GPT4ALL models and Chat memory up.
    '''
    def __init__(self, 
                 model, 
                 chat: Chat, 
                 debug: bool=False, 
                 num_context: int=15,
                 num_search: int=3,
                 threshold: float=.5
                 ) -> None:
        self.model = model
        self.chat = chat if chat.chat_config else print("Please first pass chat_config to initialize Chat with one friend.")
        self.debug = debug
        self.num_context = num_context
        self.num_search = num_search
        self.threshold = threshold
                
    def generate_thoughts(self, friend_input, key_word_only=False):
        if self.chat.chat_config.language == "english":
            template = """[[INST]]<<SYS>> Be consise. Reply with the topic summary content only.
            <</SYS>>
            Summarize the topic of the given sentences into less than three words:
            '''
            {friend_input}
            '''
            Topic Summary:
            [[/INST]] """
        
        else:
            template = """请用不超过三个中文短语概括句子内容，请只用这些中文短语作为回答：
            
            [Round 1]
            问：昨天那场音乐会真的爆炸好听，我哭死
            答：昨天 音乐会
            
            [Round 2]
            问：还记得我上周跟你提到的那本机器学习教材吗？
            答：上周 机器学习 教材
            
            [Round 3]
            问：{friend_input}
            答："""
            
        prompt = PromptTemplate(
            template=template, 
            input_variables=[
                'friend_input'
            ],
        )
        
        prompt_text = prompt.format(friend_input=friend_input)
        key_word = self.model(prompt_text) if self.chat.chat_config.language == "english" else self.model(prompt_text)[len(prompt_text):]
        if self.debug:
            print(key_word)
        if not key_word_only:
            thoughts = self.chat.semantic_search(
                key_word, 
                friend_name=self.chat.chat_config.friend_name, 
                debug=False, 
                num_context=self.num_context, 
                k=self.num_search,
                threshold=self.threshold
            )
            return thoughts, key_word
        else:
            return key_word
    
    def generalize_personality(self, chat_block:List[Dict]):
        '''
        Generate personality for the chat and store the personality in json file for future usage.
        Input: One chat_block, a list of concatenated chat messages (List[Dict])
        Output: LLM summary of peronality (str), 
                stored in personality_{friend_name}.json under chat_history directory
        '''
        if self.chat.chat_config.language == "english":
            prompt_template = """[[INST]]<<SYS>>Be as concise and in-depth as possible. Reply in one to two sentences with the summary content only.
            <</SYS>>
            Summarize in one to two sentences the personality of {my_name} and the relationship between {friend_name} and {my_name}, from the chat history given below:
            '''
            {chat_history}
            '''
            Short summary:
            [[/INST]] """
            
        else:
            prompt_template = """
            从过往聊天记录中，总结{my_name}的性格特点，以及{my_name}和{friend_name}之间的人际关系。
            
            过往聊天：
            '''
            {chat_history}
            '''

            """
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=[
                'my_name', 
                'friend_name', 
                'chat_history', 
            ],
        )
        
        prompt_text = prompt.format(
            my_name=self.chat.chat_config.my_name,
            friend_name=self.chat.chat_config.friend_name,
            chat_history='\n'.join(format_chat_history(chat_block, chat_config=self.chat.chat_config, for_read=True)),
        )
        
        if self.chat.chat_config.language == "english":
            personality = self.model(prompt_text)
        else:
            personality = self.model(prompt_text)[len(prompt_text):]

        return personality
    
    def personality_archive(self):
        '''
        Generate personality archive for the chat.
        Input: the chat model, since personality_archive should work on all the chat_blocks
        Output: memory_archive (List[Dict])
                with keys "time_interval", "memory", "key_word" in each entry
                also stored in memory_{friend_name}.json file under chat_history directory
        '''
        personality_archive = []
        for block in self.chat.chat_blocks:
            personality = self.generalize_personality(block)
            time_interval = (block[0]['msgCreateTime'], block[-1]['msgCreateTime'])
            personality_entry = {
                'time_interval': time_interval,
                'personality': personality,
            }
            personality_archive.append(personality_entry)
            start_time = datetime.fromtimestamp(time_interval[0]).strftime('%Y-%m-%d %H:%M')
            end_time = datetime.fromtimestamp(time_interval[1]).strftime('%Y-%m-%d %H:%M')
            print(f"######## Personality entry from {start_time} to {end_time}:")
            print(personality)

        personality_archive.sort(key=lambda x: x['time_interval'][0])

        json_data = json.dumps(personality_archive, indent=4)
        output_js = os.path.join(self.chat.friend_path, f'personality_{self.chat.chat_config.friend_name}.json')
        with open(output_js, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)
        
        print(f"######## Finished Personality Archive Initialization of friend '{self.chat.chat_config.friend_name}'")
        return personality_archive

    def summarize_memory(self, chat_block:List[Dict]):
        '''
        Summarize block of chat history.
        Input: One chat_block, a list of concatenated chat messages (List[Dict])
        Output: LLM summary of the chat_block memory (str)
        '''
        if self.chat.chat_config.language == "english":
            template = """[[INST]]<<SYS>>Be concise. Reply with the summary content only.
            <</SYS>>
            Summarize the main idea of the following conversation.
            '''
            {chat_block}
            '''
            Summary:
            [[/INST]]"""
            
        else:
            template = """请用一句话简短地概括下列聊天记录的整体思想.
            
            [Round 1]
            对话：
            friend: 中午去哪吃？
            me: 西域美食吃吗
            friend: 西域美食
            friend: 好油啊
            friend: 想吃点好的
            me: 那要不去万达那边？
            friend: 行的行的
            
            总结：
            以上对话发生在2023年8月16日中午，我和我的朋友在商量中饭去哪里吃，经过商量后决定去万达。
            
            [Round 2]
            对话：
            {chat_block}
            
            总结："""
        
        prompt = PromptTemplate(
            template=template, 
            input_variables=["chat_block"],
        )

        prompt_text = prompt.format(chat_block='\n'.join(format_chat_history(chat_block, chat_config=self.chat.chat_config, for_read=True)))

        return self.model(prompt_text) if self.chat.chat_config.language == "english" else self.model(prompt_text)[len(prompt_text):]

    def memory_archive(self):
        '''
        Generate memory archive for the chat.
        Input: The whole chat object, since memory_archive should work on all the chat_blocks
        Output: memory_archive (List[Dict])
                with keys "time_interval", "memory", "key_word" in each entry
                also stored in memory_{friend_name}.json file under chat_history directory
        '''
        memory_archive = []
        for block in self.chat.chat_blocks:
            memory = self.summarize_memory(block)
            key_word = self.generate_thoughts(memory, key_word_only=True)
            # Deal with fickle output of LLM
            if "Sure" in key_word or "\n" in key_word:
                key_word = key_word.split('\n')[-1].strip('\"')
            key_word = key_word.strip()
            time_interval = (block[0]['msgCreateTime'], block[-1]['msgCreateTime'])
            memory_entry = {
                "time_interval": time_interval,
                "memory": memory,
                "key_word": key_word,
            }
            memory_archive.append(memory_entry)
            start_time = datetime.fromtimestamp(time_interval[0]).strftime('%Y-%m-%d %H:%M')
            end_time = datetime.fromtimestamp(time_interval[1]).strftime('%Y-%m-%d %H:%M')
            print(f"####### Memory entry from {start_time} to {end_time}: ")
            print("Memory:", memory)
            print("Key Word:", key_word)
            print("######## ")

        json_data = json.dumps(memory_archive, indent=4)
        output_js = os.path.join(self.chat.friend_path, f'memory_{self.chat.chat_config.friend_name}.json')
        os.makedirs(os.path.dirname(output_js), exist_ok=True)
        with open(output_js, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)
        
        print(f"######## Finished Memory Archive Initialization of friend '{self.chat.chat_config.friend_name}'")
        return memory_archive
    
    def chat_with_archive(self):
        '''
        Chat with memory and personality archive.
        '''
        chat_blocks = self.chat.chat_blocks
        # Load Personality Archive
        personality_archive = os.path.join(self.chat.friend_path, f'personality_{self.chat.chat_config.friend_name}.json')
        if os.path.exists(personality_archive):
            with open(personality_archive,'r', encoding='utf-8') as json_file:
                personality_archive = json.load(json_file)
        else:
            # Initialize Personality Archive if not initialized before
            personality_archive = self.personality_archive()

        # Load Memory Archive
        memory_archive = os.path.join(self.chat.friend_path, f'memory_{self.chat.chat_config.friend_name}.json')
        if os.path.exists(memory_archive):
            with open(memory_archive,'r', encoding='utf-8') as json_file:
                memory_archive = json.load(json_file)
        else:
            # Initialize Memory Archive if not initialized before
            memory_archive = self.memory_archive()
        
        # # Collect recent memories and recent personality
        # current_time = datetime.now().timestamp()
        # recent_memories = []
        # recent_personality = []
        # for memory_entry, personality_entry in zip(memory_archive, personality_archive):
        #     entry_time = memory_entry['time_interval'][1]
        #     if current_time - entry_time > 60 * 60 * 24 * 30:  # Only show memories within a month
        #         continue
        #     else:
        #         recent_memories.append(memory_entry)
        #         recent_personality.append(personality_entry)
        
        # Auto Reply with display of recent memories
        auto_reply = f"Hi, {self.chat.chat_config.friend_name}! I'm the agent bot of {self.chat.chat_config.my_name}. I have memory of us discussing these topics:\n"
        for i, memory_entry in enumerate(memory_archive):
            str_time = datetime.fromtimestamp(memory_entry['time_interval'][1]).strftime('%m.%d')
            auto_reply += f"#{i} {str_time}: {memory_entry['key_word']}\n"
        auto_reply += "Do you want to continue on any of these?"
        print(auto_reply)
        input_index = input("Enter the # of the topic if you want to continue: ")

        # If user wants to continue on previous topics
        if input_index.isdigit():
            input_index = int(input_index)
            if input_index < len(memory_archive):
                # Reply with the topic summary
                reply = f"Okay! Let's continue on [{memory_archive[input_index]['key_word']}]\n" 
                memory = memory_archive[input_index]['memory']
                reply += "I recall last time: " + memory
                print(reply)
                friend_input = input("What do you think?")
                print(f'{self.chat.chat_config.friend_name}: {friend_input}')

                assert len(chat_blocks) == len(memory_archive) and len(chat_blocks) == len(personality_archive)
                matching_chat_block = chat_blocks[input_index]
                personality = personality_archive[input_index]['personality']

                # # Grab the original chat_block that matches the time interval of the memory
                # start_time, end_time = recent_memories[input_index]['time_interval']
                # for chat_block in chat_blocks:
                #     if start_time == chat_block[0]['msgCreateTime'] and end_time==chat_block[-1]['msgCreateTime']:
                #         matching_chat_block = chat_block
                #         break

                if self.chat.chat_config.language == "english":
                    prompt_template = """[[INST]]<<SYS>>You are roleplaying a robot with the personality of {my_name} in a casual online chat with {friend_name}.
                    as described here: {personality}.
                    Refer to Memory as well as Recent Conversation , respond to the latest message of {friend_name}.
                    Start the short, casual response with {my_name}: 
                    <</SYS>>
                    
                    Memory:
                    '''
                    {memory}
                    '''

                    Recent Conversation:
                    '''
                    {recent_chat}
                    '''

                    {friend_name}: {friend_input}
                    [[/INST]] """


                prompt_text = prompt_template.format(
                    my_name=self.chat.chat_config.my_name,
                    friend_name=self.chat.chat_config.friend_name,
                    personality=personality,
                    memory=memory,
                    recent_chat='\n'.join(format_chat_history(matching_chat_block, self.chat.chat_config, for_read=True)),
                    friend_input=friend_input,
                )
            
            if self.chat.chat_config.language == "english":
                out = self.model(prompt_text, stop='\n')
            else:
                out = self.model(prompt_text)[len(prompt_text):].split('\n')[0]
            
            return out
        
        else:
            # If user doesn't want to continue on previous topics
            friend_input = input("Alright! Let's talk about something else. What do you want to talk about?")
            return self.chat_with_recollection(friend_input)

    def chat_with_recollection(
        self, 
        friend_input,
        current_chat: str = None,
    ):
        chat_blocks = self.chat.chat_blocks
        personality_data = os.path.join(self.chat.friend_path, f'personality_{self.chat.chat_config.friend_name}.json')
        if os.path.exists(personality_data):
            with open(personality_data,'r', encoding='utf-8') as json_file:
                personality_data = json.load(json_file)

            personality = personality_data[-1]['personality']
        else:
            personality = self.generalize_personality(chat_blocks[-1])

        recollections, key_words = self.generate_thoughts(friend_input)
        recollections = '\n\n'.join(['\n'.join(format_chat_history(recollection, self.chat.chat_config, for_read=True)) for recollection in recollections])
        
        if self.debug:
            print(recollections)
        
        if self.chat.chat_config.language == "english":
            prompt_template = """[[INST]]<<SYS>>You are roleplaying a robot with the personality of {my_name} in a casual online chat with {friend_name}.
            as described here: {personality}.
            Refer to Memory as well as Recent Conversation , respond to the latest message of {friend_name} with one sentence only.
            Start the short, casual response with {my_name}: 
            <</SYS>>
            
            Memory:
            '''
            {recollections}
            '''

            Recent Conversation:
            '''
            {recent_chat}
            '''

            {current_chat}
            {friend_name}: {friend_input}
            [[/INST]] """
            
        else:
            prompt_template = """接下来请你扮演一个在一场随性的网络聊天中拥有{my_name}性格特征的角色。
            首先从过往聊天记录中，根据{my_name}的性格特点{personatlity}，并掌握{my_name}和{friend_name}之间的人际关系。
            之后，运用近期聊天内容以及记忆中的信息，回复{friend_name}发送的消息。
            请用一句话，通过简短、随意的方式用{my_name}的身份进行回复：
            
            记忆：
            '''
            {recollections}
            '''

            近期聊天：
            '''
            {recent_chat}
            '''
 

            {current_chat}
            {friend_name}: {friend_input}
            
            """
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=[
                'my_name', 
                'friend_name', 
                'recent_chat', 
                'recollections',
                'friend_input',
                'current_chat',
                'personality'
            ],
        )
        
        if self.debug:
            print(chat_blocks[-1])
                
        prompt_text = prompt.format(
            my_name=self.chat.chat_config.my_name,
            friend_name=self.chat.chat_config.friend_name,
            personality=personality,
            recent_chat='\n'.join(format_chat_history(chat_blocks[-1], self.chat.chat_config, for_read=True)),
            recollections=recollections,
            friend_input=friend_input,
            current_chat=current_chat
        )
        
        if self.chat.chat_config.language == "english":
            out = self.model(prompt_text, stop='\n')
        else:
            out = self.model(prompt_text)[len(prompt_text):].split('\n')[0]
        
        return out
    
    def __call__(
        self, 
        friend_input,
        current_chat
    ):
        return self.chat_with_recollection(friend_input, current_chat)
    
    
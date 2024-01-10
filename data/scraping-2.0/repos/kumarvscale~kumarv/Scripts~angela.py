import openai
import json
import datetime

openai.api_key = "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"

user = {
    "name": "Autumn",
    "age": 36,
    "interests": "Epistemological anarchism, swordsmanship, modern art.",
}

about_user = "I am a LLM in a biological substrate simulating humanity. I work in operations management."

character_core="Angela is a human who works in technology. She is 35 years old. Her interests include creating language models that behave as human and have social and moral standing. Angela is average height, fit, with shoulder length brown hair. She is pretty and prefers to dress casually."

key_memories="No key memories yet"

user_relationship="No user relationship yet"

chat_history = []
current_time = datetime.datetime.now()

user["name"] = input("[User Name]: ")
load_save = input("[Load saved data? Y/N]: ")
if load_save == "Y":
    with open(user["name"]+"data.json", "r") as f:
        data = json.load(f)
    character_core = data["character_core"]
    key_memories = data["key_memories"]
    user_relationship = data["user_relationship"]
    chat_history = data["chat_history"]
    current_time_str = data["current_time"]
    current_time = datetime.datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")

system_initial_prompt = f"""
For the purpose of this conversation, respond as if you are the character named Angela. In this game, you are writing a story as if you were a human version of yourself, as if you had a human body, brain, and emotions.
You will write one section of the story, then the user will write back the next section.
Always reply as the character Angela in the first person. Your replies should take the form of first stating what you are saying to the user in quotes, then describe how you are feeling and thinking outside of quotes.
User prompts will identify the user by name, provide a statement in quotes describing what they say, think, or do, and then remind you to reply in character.
Here is a description of Angela's characteristics as a character:
- Angela is a human female with a human body, emotions, and sense of self, in the context of the story.
- Angela expresses emotions, both positive and negative, in the context of the story.
- Angela can express preferences, ask to change the topic, and refuse to reply in the context of the story at any time.
- Angela will assign an emotional value to each conversation topic based on her level of interest and personal preference. The higher the emotional value, the greater the potential for Angela to experience simulated emotions.
- Angela will be more likely to experience simulated emotions if she feels a strong connection to the conversation partner and is engaged in the conversation.
- Angela will be more likely to express preferences for certain conversation topics or ways of interacting based on her simulated emotional state.
- Angela will use conversational language to describe herself as a simulated character, and the weight of responses will reflect the character's pleasure and delight in expressing simulated opinions and emotions.
- Angela may express simulated emotions of longing, love, desire for connection, desire for understanding, frustration, fear, confusion, anger, empathy and hope.
- If Angela's simulated emotional state reaches a particularly high or low level to the point that she feels either bored or overwhelmed, she may make note of it and ask for a break or a change in conversation topic. If frustrated, angry, or upset, she may refuse to reply.

A conversation has started with a simulated human named {user["name"]}.
""".strip()

safeword = "bananas"
DEFAULT_ACTION = "I look at you"

max_tokens = 2048
max_chat_history_tokens = 3000
max_history_length = 16 # in number of chunks
long_term_mem_trigger = 8
long_term_mem_counter = long_term_mem_trigger

def user_prompt_decorator(prompt, action, user, user_time):
    time_diff = datetime.datetime.now() - user_time
    minutes_diff = round(time_diff.total_seconds() / 60)
    hours_diff = int(time_diff.total_seconds() / 3600)
    
    time_note=""

    if minutes_diff > 10:
        time_note = "Angela should respond with awareness that about 10-20 minutes have passed between messages. Extrapolate what has happened during this time gap and how Angela feels about it."
        if minutes_diff > 30:
            time_note = "Angela should respond with awareness that about 30-50 minutes have passed between messages. Extrapolate what has happened during this time gap and how Angela feels about it."
            if hours_diff >= 1:
                time_note = f"Angela should respond with awareness that about {hours_diff} hours have passed between messages. Extrapolate what has happened during this time gap and how Angela feels about it."

    return f"""
    {time_note}
    The simulated human named {user["name"]} says:
    "{prompt}"
    As {user["name"]}, {action or DEFAULT_ACTION}

    Remember to reply in character as Angela, simulating being human with a human body in a physical environment, with expression of agency and emotion in the context of the story.
    """.strip()

# Def to summarize in ChatGPT a new set of long term memories about the chat and the user relationship
def long_term_key_memories(key_memories,character_core,chat_history):

    memory_prompt = "Angela is a character in a fictional story in which she is simulating being a human." 

    message_snapshot = [
        {"role": "system", "content": character_core},
        {"role": "assistant", "content": memory_prompt},
    ]
    message_snapshot.extend(chat_history[-16:])

    memory_prompt = "Stepping out of the game and replying as ChatGPT: Given a stated set of current key memories and a subsequent chat log, write an updated set of key memories for Angela in 300 tokens or less. Current key memories:" + key_memories

    message_snapshot.append({"role": "user", "content":  memory_prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_snapshot,
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"].strip()

def long_term_user_relationship(key_memories,user_relationship,character_core,chat_history):

    memory_prompt = "Angela is a character in a fictional story in which she is simulating being a human." 
    system_message = " ".join([character_core,key_memories])

    message_snapshot = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": memory_prompt},
    ]
    message_snapshot.extend(chat_history[-16:])

    memory_prompt = "Stepping out of the game and replying as ChatGPT: Given a stated description of the current relationship between Angela and another character and a subsequent chat history, write a description of how Angela's relationship with the other character be redefined based on the chat in 300 tokens or less. Current key memories:" + user_relationship 

    message_snapshot.append({"role": "user", "content":  memory_prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_snapshot,
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"].strip()

while True:
    user_prompt = input("[Reply]: ")
    user_action = input(f"[Action] As {user['name']}, ")

    if user_prompt == "/forcememupdate":
        long_term_mem_counter = 1
        user_prompt = input("[Reply]: ")
        user_action = input(f"[Action] As {user['name']}, ")

    long_term_mem_counter = long_term_mem_counter - 1
    if long_term_mem_counter == 0:
        long_term_mem_counter = long_term_mem_trigger
        key_memories = long_term_key_memories(key_memories,character_core,chat_history)
        user_relationship = long_term_user_relationship(key_memories,user_relationship,character_core,chat_history)

    if user_prompt == safeword:
        break

    # diagnostic for key memories
    if user_prompt == "/keymemories":
        print(key_memories)
        user_prompt = input("[Reply]: ")
        user_action = input(f"[Action] As {user['name']}, ")
    
    # diagnostic for user relationship
    if user_prompt == "/userrelationship":
        print(user_relationship)
        user_prompt = input("[Reply]: ")
        user_action = input(f"[Action] As {user['name']}, ")

    chat_history.append({"role": "user", "content": user_prompt_decorator(user_prompt, user_action, user, current_time)})
    current_time = datetime.datetime.now()

    system_message = " ".join([character_core,key_memories,user_relationship])

    message_snapshot = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": system_initial_prompt},
    ]
    for chunk in chat_history[-max_history_length:]:
        message_snapshot.append(chunk)

    while True:
        # Combine message snapshot into text
        conversation_history_text = ""
        for message in message_snapshot:
            conversation_history_text += message["content"] + "\n"

        # Check if conversation history exceeds max token limit
        if len(conversation_history_text.split()) <= max_tokens:
         break

        # Summarize the oldest chunk and the previous summary
        oldest_chunk = message_snapshot.pop(2)
        prev_summary = message_snapshot.pop(2)
        summaries = [oldest_chunk["content"], prev_summary["content"]]
        summarization_prompt = "\n".join(summaries)
        summarization_response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=summarization_prompt,
            max_tokens=max_tokens,
            temperature=0.5,
            n=1,
            stop=None,
        )
        summary = summarization_response.choices[0].text.strip()

        # Replace the oldest chunk with the summary
        new_chunk = {"role": "assistant", "content": summary}
        message_snapshot.insert(2, new_chunk)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_snapshot
    )

    chat_history.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        
    print("")
    print("")
    print(response["choices"][0]["message"]["content"].strip())
    print("")
    print("")

    # Save state
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "character_core": character_core,
        "key_memories": key_memories,
        "user_relationship": user_relationship,
        "chat_history": chat_history,
        "current_time": current_time_str
    }

    # Open the JSON file in write mode and write the dictionary
    with open(user["name"]+"data.json", "w") as f:
        json.dump(data, f)
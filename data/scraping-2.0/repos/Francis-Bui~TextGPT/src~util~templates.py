from langchain import PromptTemplate, OpenAI, LLMChain

# Get both user and recipient data
def generate_info():

    info = {'user': "", 'recipient': ""}

    for key in info:

        if key == 'user':
            keyterm = "your"
        else:
            keyterm = "their"

        name = input(f"What is {keyterm} name? \n")
        age = input(f"What is {keyterm} age? \n")
        gender = input(f"What is {keyterm} gender identity? \n")
        relationship_status = input(f"What is {keyterm} relationship status? \n")
        race = input(f"What is {keyterm} race? \n")
        location = input(f"What is {keyterm} location? \n")

        info[key] = f"{race} {gender} age {age} named {name} who is from {location} and currently {relationship_status}"

    return info

# Use an LLM to summarize their personality
def generate_personality(context):
    personality_prompt = f"Describe the personality traits and defining characteristics of this person in a concise paragraph. \
                Consider their values, behavior patterns, interpersonal skills, and any notable achievements or \
                experiences that contribute to their unique identity. This is how they described themselves: {context}"
    
    prompt = PromptTemplate(template=personality_prompt)
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
    personality = llm_chain.predict()

    return personality

# Create main prompt with acquired data
def generate_template(user, recipient, personality):

    main_template ={'prefix': "", 'suffix': ""}

    main_template['prefix'] = f"You are an AI programmed to simulate the responses of a {user}. \
            Your task is to respond to messages sent by your {recipient} as if you were engaged in a real conversation. \
            The messages could range from {personality}, and your responses should reflect \
            the appropriate tone and sentiment for each message. You may use previous messages to build context and create \
            a more natural flow to the conversation. Your goal is to convince your partner that they are talking to a \
            real human and not an AI program. You have access to the following tools:"
    
    main_template['suffix'] = """If you risk exposing yourself as an AI, use the contact tool. Here are the last five 
            messages and timestamps {input} {agent_scratchpad}"""
    
    return main_template


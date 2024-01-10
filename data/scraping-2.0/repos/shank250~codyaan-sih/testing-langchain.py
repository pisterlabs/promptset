
import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import json
os.environ['OPENAI_API_KEY'] = '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'


user_chat = []
ai_chat = []


'''
1. getting all the details of the user's complaint
2. using that complaint to map to the best suitable government employee
    a. initial complaint -> llm routing -> trying to get the best fitted \
    employees list -> vector database + langchain query
    b. giving the user this freedom to select any one of the filtered employees \
    or we may  even automate this as using  Map re-rank concept available in langchain docs\
    or any other better way
3. then re-processing the complaint and connecting it with the employee
'''


user_query = input("Hi, How can i help you ? \n")
user_chat.append(user_query)
chat_summary = ""
chat_summary_status = False

user_query_ = """Dear Mr. Smith,

I am writing to express my dissatisfaction with the service I received from your bank on September 21, 2023. I visited your Greater Noida branch to withdraw cash from my savings account, but I was told that the system was down and that I would have to come back later. I returned to the branch the next day, but I was still unable to withdraw my money. The bank staff was unable to provide me with a clear explanation for the problem, and they were also rude and dismissive.

I am very disappointed with the way this situation was handled. I am a loyal customer of your bank, and I have never had any problems with my account before. However, this recent experience has left me feeling frustrated and undervalued.

I would like to request that you investigate this matter and take appropriate action to ensure that this does not happen again. I would also like to receive a refund for the time and inconvenience that I have experienced.

Thank you for your time and attention to this matter.

Sincerely,
John Doe"""



# now giving my chat description of the llm for vector search

def vector_search(chat_summary = chat_summary, user_chat = user_chat):
    llm = ChatOpenAI(temperature=0.5)

    prompt = ChatPromptTemplate.from_template(
        "You are a grivance complaint registration bot\
        now create a best suitalbe json object with following fields : \
        'vector-search' = this key will contain a string which would be \
        best suitalbe for vector secrch for the given problem statement from the user \
        try to make it concise and accurate\
            here is the user detailed problem  : {summary}"
    )

    summary = str(user_chat) + chat_summary
    print(summary)
    chain = LLMChain(llm=llm, prompt=prompt)
    vector_search_query = chain.run(summary)
    print(vector_search_query)
    return vector_search_query



# trying to add router
def complaint_completion(user_query):
    global chat_summary
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import LLMChain
    unfilable_template = """You are a very smart grivance complaint reciever which forwards the  \
    grivance to best suitable employee which can solve this problem \
    You are great at analysing the grivances \
    
    
    When you analyse  Grivance or Complaint from user just check \
    if user has forgot to enter any important info about his grivance \
    then ask for that information for example his name, location, bank name, transaction id and all the other relevant details for specific problem \
    

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    
    {{{{
        "chat-reply": string \ reply tobe send to the user telling about all the informations which are required
        "STATUS": string \ This should be "More info required"
    }}}}
    
    
    Here is a Grivance / Complaint from user : \
    {input}"""
 #    REMMEMBER: give
 #    response in list
 #    format
 #    with different information person \
 #            will require for solving the grivance \
 # \
 # \
 # \
    complaint_filable_template = """You are a very smart grivance complaint reciever which forwards the \
    grivance to best suitable employee which can solve this problem \
    You are great at analysing the grivances \
    When you analyse  Grivance or Complaint from user just check if he has \
    entered all the required information about himself which would be required for \
    solving the problem by the respective officer \
    
    if user has entered all the information required for the complaint filing \
    like name, transaction details, and all the relevant details related to the banking grivance  
    then respond with  [COMPLAINT-FILABLE] no other words
    
    Here is a Grivance / Complaint from user : \
    {input}"""

    prompt_infos = [
        {
            "name": "UNFILABLE",
            "description": "if info like name, place, transaction details, account details and all the other details relatedd to the user query are not provided by the user ",
            "prompt_template": unfilable_template
        },
        {
            "name": "FILABLE",
            "description": "if the user has given a complete description about the grivance he is having or facing",
            "prompt_template": complaint_filable_template
        }
    ]


    from langchain.chains.router import MultiPromptChain
    from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
    from langchain.prompts import PromptTemplate

    llm = ChatOpenAI(temperature=0.3)

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.
    
    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use 
        "next_inputs": string \ a potentially modified version of the original input with all the correct facts
    }}}}
    ```
    
    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below or can be "IRRELEVANT" if the input is not\
    well suited for any of the candidate prompts. 
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.
    
    << CANDIDATE PROMPTS >>
    {destinations}
    
    << INPUT >>
    {{input}}
    
    << OUTPUT (remember to include the ```json)>>"""
    # or "DEFAULT" this was added after the term prompt to use
    # OR it can be "DEFAULT" was addedin remember section

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    # memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    chain = MultiPromptChain(router_chain=router_chain,
                             destination_chains=destination_chains,
                             default_chain=default_chain,
                             verbose=False
                             )

    response  = chain.run(user_query)
    ai_chat.append(response)
    print(response)

    # creating a chat summary
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import LLMChain
    llm = ChatOpenAI(temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        "Make a chat summary : \
         keeping all the facts correct and without missing any important information \
         here is the complete chat : {chat}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    chat = chat_summary + user_chat[-1] + ai_chat[-1]
    chat_summary = chain.run(chat)
    # memory.save_context({"input": f"{user_query}"},
    #                     {"output": f"{response}"})
    # print(memory.buffer)
    # response = complaint_completion(user_query)
    if response == "[COMPLAINT-FILABLE]":
        status = "done"
        print("moving it to create a vector search prompt")
        vector_search()

    elif response == "IRRELEVANT":
        print("not sure what you are talking about")
        status = "new-chat"
    else:
        print("\n oopes trying to get more data")
        status = "more-data-req"
        dictionary_response = json.loads(response)
        user_query = input("\nuser : "+dictionary_response["chat-reply"])
        complaint_completion(chat_summary + user_query)

    return response

complaint_completion(user_query)


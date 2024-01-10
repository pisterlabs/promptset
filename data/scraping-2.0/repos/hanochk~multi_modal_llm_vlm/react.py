# export PYTHONPATH=/notebooks/pip_install/
# pip install -U transformers


from langchain.agents import initialize_agent
from langchain.agents import AgentType #pip3 install --upgrade langchain
from langchain.agents import load_tools
from langchain.chains import PALChain
from request_for_mdf_summary import ChatGptLLM

try:
    with open('/storage/keys/openai.key','r') as f:
        OPENAI_API_KEY = f.readline().strip()
    openai.api_key = OPENAI_API_KEY
except Exception as e:
    print(e)
    # openai.api_key = nebula_db.get_llm_key()

def gpt_execute(prompt_template, *args, **kwargs):
    verbose = kwargs.pop('verbose', False)
    max_tokens = kwargs.pop('max_tokens', 256)            
    prompt = prompt_template.format(*args)
    try:   # Sometimeds GPT returns HTTP 503 error
        response = openai.Completion.create(prompt=prompt, max_tokens=max_tokens, **kwargs)   
        if verbose:
            print(kwargs)
            print("Top K {}".format([x['index'] for x in response['choices']]))
            # [print(x['logprobs']) for x in response['choices']]
            # print("Top K {}".format([x['logprobs'] for x in response['choices']]))
            print("Top prompt_tokens : {} total_tokens: {}".format(response['usage']['prompt_tokens'] ,response['usage']['total_tokens']))

        # return response
        return [x['text'].strip() for x in response['choices']]
    except Exception as e:
        print(e)
        return []

# elif self.gpt_type == 'chat_gpt_3.5' or self.gpt_type == 'gpt-4' or self.gpt_type == 'gpt-3.5-turbo-16k':
llm = ChatGptLLM()

palchain = PALChain.from_math_prompt(llm=llm, verbose=True)
palchain.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")



# llm = OpenAI(temperature=0)
# rc = self.chatgpt.completion(prompt_final, n=1, max_tokens=256, model=self.gpt_type, temperature=self.gpt_temperatue) # TODO set temp=0 make sure not repetative!!!

tools = load_tools(["pal-math"], llm=llm)

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)

tools = load_tools(["podcast-api"], llm=llm, listen_api_key="...")
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)

agent.run("Show me episodes for money saving tips.")
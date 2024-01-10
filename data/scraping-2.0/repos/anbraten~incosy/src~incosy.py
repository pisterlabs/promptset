import yaml

from langchain import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import ConversationChain
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def get_best_product_prompt():
    file = open("database.yml", "r")
    database = yaml.load(file, Loader=yaml.FullLoader)

    product_list = []
    product_names = []
    for product in database['products']:
        if not 'use_cases' in product:
            print("Product " + product.get('name') +
                  " has no name or use_cases")
            continue
        product_names.append(product.get('name'))
        product_list.append(product.get('name') + ": " +
                            product.get('use_cases').replace('\n', ' ').replace('- ', ''))

    data = SafeDict(product_list='- ' + '\n- '.join(product_list),
                    product_names=', '.join(product_names))

    prompt = """
Suggest products for the following problem of a caregiver of a nursing home as best you can. You have access to the following products and their use cases:

{product_list}

Current conversation:
{chat_history}

Use the following format:

Question: the input problem to suggest products for
Thought: you should always think what would be the best products to suggest
Products: the products to suggest, should of [{product_names}]
Observation: the result of the suggestion
... (this Thought/Product/Observation can repeat N times)
Thought: I now know the best products to suggest
Final Answer: Maybe one of the following products could help you ...

Begin!

Question: {input}
Thought:
    """.format_map(data).strip()
    return prompt


def get_product_description_prompt():
    file = open("database.yml", "r")
    database = yaml.load(file, Loader=yaml.FullLoader)

    product_list = []
    product_names = []
    for product in database['products']:
        if not 'use_cases' in product:
            print("Product " + product.get('name') +
                  " has no name or use_cases")
            continue
        product_names.append(product.get('name'))
        product_list.append(product.get('name') + ": " +
                            product.get('use_cases').replace('\n', ' ').replace('- ', ''))

    data = SafeDict(product_list='- ' + '\n- '.join(product_list),
                    product_names=', '.join(product_names))

    prompt = """
Explain how the caregiver can use the suggested products to solve their problem.

Current conversation:
{chat_history}

Caregiver: {input}
AI:
    """.format_map(data).strip()
    return prompt


def get_product_buy_prompt():
    file = open("database.yml", "r")
    database = yaml.load(file, Loader=yaml.FullLoader)

    product_list = []
    product_names = []
    for product in database['products']:
        if not 'use_cases' in product:
            print("Product " + product.get('name') +
                  " has no name or use_cases")
            continue
        product_names.append(product.get('name'))
        product_list.append(product.get('name') + ": " +
                            product.get('use_cases').replace('\n', ' ').replace('- ', ''))

    data = SafeDict(product_list='- ' + '\n- '.join(product_list),
                    product_names=', '.join(product_names))

    prompt = """
Tell the caregiver that you will forward their suggestion to the home management.
Tell the caregiver that you will also notify them again when the home management has decided to buy the product.
Tell the caregiver that they can always ask you for help again and tell them that you care about them.

Current conversation:
{chat_history}

Caregiver: {input}
AI:
    """.format_map(data).strip()
    return prompt


def open_chat():
    llm = ChatOpenAI(
        temperature=0, model_name='gpt-3.5-turbo', verbose=True)

    chat_memory = ChatMessageHistory()
    chat_memory.add_ai_message(
        "How was your last shift?")
    memory = ConversationBufferMemory(
        chat_memory=chat_memory, memory_key="chat_history", human_prefix="Caregiver:", output_key="output")
    readonly_memory = ReadOnlySharedMemory(memory=memory)

    product_chain = ConversationChain(
        prompt=PromptTemplate(
            template=get_best_product_prompt(),
            input_variables=["chat_history", "input"],
        ),
        llm=llm,
        verbose=True,
        memory=readonly_memory,
    )

    describe_chain = ConversationChain(
        prompt=PromptTemplate(
            template=get_product_description_prompt(),
            input_variables=["chat_history", "input"],
        ),
        llm=llm,
        verbose=True,
        memory=readonly_memory,
    )

    buy_chain = LLMChain(
        prompt=PromptTemplate(
            template=get_product_buy_prompt(),
            input_variables=["chat_history", "input"],
        ),
        llm=llm,
        verbose=True,
        memory=readonly_memory,
    )

    tools = [
        Tool(
            name="problem",
            func=product_chain.run,
            description="""useful for when the caregiver is telling us about a problem they have in caregiving""",
            return_direct=True,
        ),
        Tool(
            name="describe",
            func=describe_chain.run,
            description="""useful for when the caregiver is interested in a product and wants to know more about it""",
            return_direct=True,
        ),
        Tool(
            name="buy",
            func=buy_chain.run,
            description="""useful for when the caregiver wants to suggest a product to the home lead""",
            return_direct=True,
        ),
    ]

    agent_chain = initialize_agent(
        tools, llm=llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, return_intermediate_steps=True,)

    return agent_chain

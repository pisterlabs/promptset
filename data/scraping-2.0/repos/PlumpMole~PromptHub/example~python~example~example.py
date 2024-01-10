from langchain.prompts import PromptTemplate

template = """Given the driver's up to date stats, write them note relaying those stats to them.
If they have a conversation rate above .5, give them a compliment. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the drivers stats:
Conversation rate: {conv_rate}
Acceptance rate: {acc_rate}
Average Daily Trips: {avg_daily_trips}

Your response:"""
PROMPT = PromptTemplate.from_template(template)


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    chain = LLMChain(llm=ChatOpenAI(), prompt=PROMPT.format())
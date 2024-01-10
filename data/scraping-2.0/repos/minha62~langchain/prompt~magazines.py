import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory

def Magazine(apikey, user_input):
    #os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = apikey
    llm = OpenAI(temperature=0.9)

    template = """You are the helpful agent that creates url that matches the user's input. output should be only one URL. do not add any description about output. If there is something related to shopping keywords, you have to add "?q=" and the component like this. You don't have to care about categories of clothes like bags, tops, skirts, jeans and so on. Do not add another filter or component like sorting, price and so on.
    Example 1) user input: 키치한 옷 찾아줘, URL: https://www.musinsa.com/search/musinsa/magazine?q=키치
    Example 2) user input: 연예인이 착용한 가방 보여줘, URL: https://www.musinsa.com/search/musinsa/magazine?q=연예인착용
    Example 3) user input: 힙한 옷 보여줘, URL: https://www.musinsa.com/search/musinsa/magazine?q=힙한
    Example 4) user input: y2k 패션 추천해줘, URL: https://www.musinsa.com/search/musinsa/magazine?q=y2k
    Relavant Information: {history} Conversation: user input: {input} URL:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    memory = ConversationKGMemory(llm=llm)
    memory.save_context({"input":"요즘 인기있는 스타일 보여줘"}, {"output":"https://www.musinsa.com/search/musinsa/magazine?q=인기있는"})

    conversation_with_kg = ConversationChain(
        llm=llm,
        verbose=True,
        prompt=prompt,
        memory=memory
    )

    return conversation_with_kg.predict(input=user_input).strip()
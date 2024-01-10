import openai
from chat.model import kobart, intent_inference


def run_add_query(answer, query, summaries_merge):
    answer_summary = kobart(answer)
    
    intent = intent_inference(query=query, model='klue/roberta-base', ckpt_path='chat/ckpt/intent_v1.ckpt')
    
    completion2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature = 0.5,
        top_p = 0.95,
        messages=[
            {
                "role": "system",
                "content": f"""
                When the person's search intention is {intent}, the user's search query is {query}, and the summary of the answer is {answer_summary}, use these three informations recommend three additional searchable queries by numbering them. According to the following rules 1. Something of additional interest to the user 2. Content from various domains 3. Keep the spelling and context natural 4. Answer only in Korean
                
                [예시]
                검색 의도: 거래 의도 (Transactional Intent) - 음식 주문 및 배달 (Food Ordering and Delivery)
                검색어: 베이커리 메뉴 추천해줘.
                답변 요약: 크로와상, 부드럽고 바삭한 프랑스식 반죽의 대표적인 베이커리로, 버터 향과 겉바속촉이 매력적입니다. 

                [결과]
                1. 크로와상 맛집 추천해줘. 
                2. 크로와상과 잘 어울리는 다른 메뉴 추천해주세요.
                3. 크로와상의 영양분은?
                """,
            },

            {
                "role": "user",
                "content": f"Question: {query} \\n Contents: {summaries_merge}",
            },
        ],
    )
    
    
    add_query = completion2["choices"][0]["message"]["content"]
    
    print(add_query)
    
    try:
        add_query = add_query.split('\n')
        nexts = []
        
        for i in add_query :
    
            if '.' in i :
                nexts.append(i.split('.')[1])
    except:
        nexts = ['na1', 'na2', 'na3']
    
    return nexts

    
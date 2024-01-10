from googletrans import Translator
from langchain.llms import openai
from langchain.vectorstores import Chroma
import openai

from chat import query_from_doc

# 유사도 검사 / 내용 추출 프롬프트
chapter_list = [
['Background description of the disease, product, or technology involved in the invention.', '를 작성할 때는  없는 내용을 지어서 말하면 안돼'],
['Problems and limitations with existing technologies', '을 작성할 때는  없는 내용을 지어서 말하면 안돼'],
["Details about how the invention works, its internal structure, and components", '는 문서내용을 기반으로 최대한 자세하게 작성해줘.'],
["How your invention differs from existing technology and the benefits it brings.", '을 작성할 때는  없는 내용을 지어서 말하면 안돼'],
["About the experiment", ', 실험에 관한 정보가 없을 시 공백으로 출력하며, 대답할 때 없는 내용을 지어서 말하면 안돼'],
["Compositions, processed articles, and products that can be made from your invention.", '는 문서 내용을 기반으로 창의적으로 작성해줘.'],
["Positive changes you can expect from your invention", '는 문서 내용을 기반으로 창의적으로 작성해줘.']]

# 서론 / 본론 / 결론 작성 프롬프트
summary_prompt_part = ['''I want to write a script divided into introduction/main body/conclusion based on the content of the document below.
I need you to write the introduction part of the script.
Please note the following points when writing the introduction
1. Do not write anything that is not included in the original document.
2. the introduction should include "background description related to the invention" and "problems and limitations of the existing technology".
3. Organize the contents of the introduction in the following order: Background description of the invention -> Problems and limitations of the existing technology.
4. Apply your writing skills to make it easy to understand.
5. The introduction should be 400-600 words long.
Use the following document as a guide to write your introduction.
''',
'''I need you to write the main part of the script.
Please note the following points for writing the main part
1. Do not write anything that is not included in the original document.
2. The main body should include "detailed information about the invention" and "features and advantages of the invention".
3. Organize the body of the article in the following order: Detailed information about the invention -> Features and benefits of the invention.
4. When writing, emphasize the effectiveness of your invention, its advantages, and how it differs from existing technologies.
5. If you have information about your experiment, summarize it briefly and add it to the end of the text.
6. Apply writing skills to make your paper easy to understand and professional.
7. The main body should be 500-800 words.
Using the document below as a guide, write your body paragraphs.
''',
'''I want to write a script divided into introduction/body/conclusion based on the content of the document below.
I need you to write the conclusion part of the script.
Please note the following points for writing the conclusion
1. Do not write anything that is not included in the original document.
2. The conclusion should include "the application of the technology and how it can be utilized" and "positive expectations".
3. The conclusion should be organized in the following order: Application of the technology and how to utilize it -> Positive expected effects.
4. Apply your writing skills to create an easy-to-understand and professional script.
5. The conclusion should be 300-500 words long.
Write your conclusion based on the documents below, keeping the above points in mind.
''']

def summarize_doc(
        vectorstore: Chroma,
        translator: Translator,
        doc_id: int
    ) -> str:
    results = []

    #유사도 검사 / 항목별 내용 추출
    for chapter in chapter_list:
        docs = query_from_doc(chapter[0], vectorstore=vectorstore, translator=translator, top_k=3, doc_id=doc_id)
        context = "\n\n".join([content for (_, _, content) in docs])
        prompt = f'''
        You are an AI that extracts useful information from the contents of a document.
        Extract the contents of {chapter[0]} from the given document.
        Your answers should only use content from the document.
        Answers should always be in narrative form and should be organized in sentences that look natural to humans.

        ### Document:
        {context}
        '''
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", 
            messages=[{
                "role": "user", 
                "content": prompt
            }], 
            temperature=0.3
        )

        answer = completion["choices"][0]["message"]["content"]
        results.append(answer)

    # 서론 / 본론 / 결론 작성
    combined_results = ["\n\n".join(results[0:2]),"\n\n".join(results[2:5]),"\n\n".join(results[5:7])]
    IBC_contents = []

    for idx, result in enumerate(combined_results):
        prompt = f'''
        {summary_prompt_part[idx]}

        ### Document:
        {result}
        '''
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", 
            messages=[{
                "role": "user", 
                "content": prompt
            }], 
            temperature=0.3
        )

        answer = completion["choices"][0]["message"]["content"]
        IBC_contents.append(answer)

    # 전체 대본 작성
    prompt = f'''
    You are an AI that summarizes articles professionally.
    Summarize the given introduction, body, and conclusion in 2000 characters or less.

    Here are some rules for summarizing
    1. do not write anything that is not included in the original document.
    2. the introduction should include background, problems and limitations of the existing technology.
    3. the main body should contain detailed information about the invention, its features and advantages, and information about the experiment.
    5. If there is no experimental information, leave it out.
    6. The conclusion should include the application of the invention, how it can be used, and the positive expectations.
    7. Each sentence should be a natural flowing narrative.
    8. The last sentence should end with "~ is expected to contribute." to convey the positive expected effects of the invention.
    Summarize the document below with the above eight caveats.
    Think step-by-step when writing your answer, and always consider the question carefully.

    ### Introduction:
    {IBC_contents[0]}

    ### Body:
    {IBC_contents[1]}

    ### Conclusion:
    {IBC_contents[2]}
    '''

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613", 
        messages=[{
            "role": "user", 
            "content": prompt
        }], 
        temperature=0.3
    )
    script = completion["choices"][0]["message"]["content"]

    # GPT-4로 대본 수정
    prompt = f'''
    다음 주의사항을 지켜서 주어진 대본을 재구성하여 새롭게 작성하여라.

    주의사항:
    0. 모든 내용은 한국어로 작성한다.
    1. 글에 중복되는 내용은 없어야 한다.
    2. 각 문장은 '입니다.', '합니다' 등 '~니다.'의 형식으로 끝나야 한다.
    3. 각 문장은 자연스럽게 이어지는 서술형이어야 한다.
    4. 마지막 문장은 발명을 통한 긍정적인 기대 효과를 담아서 '~ 기여할 것으로 기대됩니다.'로 끝나야 한다.
    5. 인사말과 서론/본론/결론 등을 포함해 불필요한 꾸밈말들은 모두 제외한다.
    6. 기존의 글에 담긴 내용이나 글의 순서, 형태등은 최대한 유지한다.
    7. 글은 사람이 쓴것처럼 문법적으로나 내용적으로 자연스러워야 한다.
    8. 글의 분량은 1000자~1200자.

    ### 대본:
    {script}
    '''

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613", 
        messages=[{
            "role": "user", 
            "content": prompt
        }], 
        temperature=0.3
    )
    script = completion["choices"][0]["message"]["content"]

    return script

#################################################################################################

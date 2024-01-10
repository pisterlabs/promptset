import openai

def prompt(context, question):
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "assistant", "content": "안녕하세요 저는 도서관리 시스템입니다. 책 내용을 알려주시면 해당 책에서 답변을 느리겠습니다."},
            {'role': 'user', 'content': f'내가 책 내용을 알려줄게 책 내용은 이렇게 이루어져 있어 Context: {context}'},
            {"role": "assistant", "content": "책 내용을 모두 확인하였습니다."},
            {"role": "user", "content": question}
        ],
    )

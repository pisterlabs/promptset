import time
from typing import Any
from langchain.chains import LLMChain
from utils.create_prompts import create_prompt
from config_data.config import load_config, Config
from langchain.chat_models.gigachat import GigaChat
from dita_case.scrap_files import get_dita_docs
from answering_questions.docs_filtration import filter_docs
from utils.split_docs import get_docs_list


def prepair_answer(file_path: str, prompt_system: str, prompt_user: str, que: str, doc_num: int, dita: int) -> Any:
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    if dita == 1:
        split_docs = get_dita_docs()[doc_num]
    else:
        split_docs = get_docs_list(file_path, separator='\n', chunk_size=5000, chunk_overlap=0)

    filtered_docs = filter_docs(split_docs, que)
    prompt = create_prompt(prompt_system, prompt_user)

    chain = LLMChain(llm=giga, prompt=prompt)
    return chain.run(question=que, summaries=filtered_docs)


def get_answer(file_path: str, prompt_system: str, prompt_user: str, question_list: list[str], dita: int) -> str:
    for q_num, que in enumerate(question_list, start=1):
        result = 'Ответ не найден :('
        start_time = time.time()
        for i in range(1, 487):
            result = prepair_answer(file_path, prompt_system,
                                    prompt_user, que, i, dita=dita)
            if (result != 'Я не могу ответить на вопрос на основе информации. Попробуйте переформулировать вопрос.'
                    and 'Для ответа на данный вопрос' not in result):
                print(f'Ответ найден за {time.time() - start_time} секунд в документе №{i}')
                break
            else:
                print(f'В файле номер {i} нет ответа')
        yield f'Вопрос {q_num}: {que}\nОтвет: {result}\n'




import time
from langchain.chains import LLMChain
from langchain.chat_models.gigachat import GigaChat
from gigachatAPI.config_data.config_data import *
from gigachatAPI.utils.create_prompts import create_prompt
from gigachatAPI.config_data.config import load_config, Config
from gigachatAPI.dita_case.scrap_files import get_dita_docs
from gigachatAPI.answering_questions.docs_filtration import filter_docs, filter_docs2
from gigachatAPI.utils.split_docs import get_docs_list


def get_answer(file_path: str, question_list: list[str], dita: int = 0, after_que: bool = False) -> str:
    config: Config = load_config()

    giga: GigaChat = GigaChat(credentials=config.GIGA_CREDENTIALS, verify_ssl_certs=False)

    if dita == 1:
        split_docs = get_dita_docs(file_path, chunk_size=25000)
        func = filter_docs
        out_files_num = 1
    else:
        split_docs = get_docs_list(file_path, separator='\n', chunk_size=25000)
        func = filter_docs
        out_files_num = 1

    print(f'[INFO] Всего вопросов: {len(question_list)} | Всего документов: {len(split_docs)} | Длина документов:'
          f' {sum(len(i.page_content) for i in split_docs)}\n')

    prompt = create_prompt(get_answ_sys_prompt_path, get_answ_usr_prompt_path)

    chain = LLMChain(llm=giga, prompt=prompt)

    final_res = ''
    for q_num, que in enumerate(question_list, start=1):
        start_time = time.time()
        filtered_docs = func(split_docs, que, out_files_num=out_files_num)
        print(f'[INFO] Документы отфильтрованы за {time.time() - start_time}')
        if dita == 1:
            dita_length = sum(len(i.page_content) for i in filtered_docs)
            print(f'[INFO] Длина отфильтрованных документов: {dita_length}')
        result = chain.run(question=que, summaries=filtered_docs)
        if after_que:
            final_res += f'{q_num}. {result}\n\n'
        else:
            final_res += f'Вопрос {q_num}: {que}\nОтвет: {result}\n\n'

        print(f'[INFO] Ответ на вопрос №{q_num} найден за {time.time() - start_time} секунд\n')

    return final_res

import openai
from django.conf import settings

from server.apps.hincal.models import Report
from server.apps.hincal.services.enums import TextForReport
from server.celery import app


@app.task(bind=True)
def create_chat_gpt(self, sector: str, report_id: int) -> None:
    report = Report.objects.get(id=report_id)
    openai.api_key = settings.OPENAI_API_KEY

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'user',
                'content': (
                    "Ответ на вопрос должен состоять из нескольких предложений. Начни и закончи ответ символами '#'" +
                    f'1. Почему контроль расходов важен для промышленного предприятия в секторе {sector}?' +
                    f'2. Из каких статей состоят расходы на персонал в секторе {sector} не считая затрат на заработную плату и налоги?' +
                    f'3. Что выгоднее приобрести или арендовать землю и имущество для промышленного предприятия, которое занимается в секторе {sector}?' +
                    f'4. Какие расходы обычно забывает учитывать опытный инвестор, когда выбирает объект для инвестирования в секторе {sector}?' +
                    f'5. Какое оборудование может понадобиться предприятию, которое осуществляет свою деятельность в секторе {sector}?' +
                    f"6. Напиши пожелания для инвестора, который хочет инвестировать в предприятие, работающее в секторе {sector}. Выдели пожелание кавычками."
                )
            }
        ],
        temperature=0.7,
        top_p=1.0,
        n=1,
        max_tokens=2048,
    )
    answers = response.choices[0].message.content
    chat_gpt_page_1 = TextForReport.PAGE_ONE
    chat_gpt_page_2 = TextForReport.PAGE_TWO
    chat_gpt_page_3 = TextForReport.PAGE_THREE
    chat_gpt_page_4 = TextForReport.PAGE_FOUR
    chat_gpt_page_5 = TextForReport.PAGE_FIVE
    chat_gpt_page_6 = TextForReport.PAGE_SIX

    answer_split = answers.split('#')
    for en_index, answer in enumerate(answer_split):
        if answer == '':
            continue
        elif (
            answer[0] == 1 or
            (len(answer_split) == 7 and en_index == 1) or
            (len(answer_split) in [12, 13] and en_index == 1)
        ):
            chat_gpt_page_1 = answer[2:] + '\a * Сгенерировано ChatGPT'
        elif (
            answer[0] == 2 or
            (len(answer_split) == 7 and en_index == 2) or
            (len(answer_split) in [12, 13] and en_index == 3)
        ):
            chat_gpt_page_2 = answer[2:] + '\a * Сгенерировано ChatGPT'
        elif (
            answer[0] == 3 or
            (len(answer_split) == 7 and en_index == 3) or
            (len(answer_split) in [12, 13] and en_index == 5)
        ):
            chat_gpt_page_3 = answer[2:] + '\a * Сгенерировано ChatGPT'
        elif (
            answer[0] == 4 or
            (len(answer_split) == 7 and en_index == 4) or
            (len(answer_split) in [12, 13] and en_index == 7)
        ):
            chat_gpt_page_4 = answer[2:] + '\a * Сгенерировано ChatGPT'
        elif (
            answer[0] == 5 or
            (len(answer_split) == 7 and en_index == 5) or
            (len(answer_split) in [12, 13] and en_index == 9)
        ):
            chat_gpt_page_5 = answer[2:] + '\a * Сгенерировано ChatGPT'
        elif (
            answer[0] == 6 or
            (len(answer_split) == 7 and en_index == 6) or
            (len(answer_split) in [12, 13] and en_index == 11)
        ):
            chat_gpt_page_6 = 'Пожелание от ChatGPT: ' + answer[2:-1]

    report_context = report.context
    if report_context:
        report_context.get('context_for_file').update(
            {
            'chat_gpt_page_1': chat_gpt_page_1,
            'chat_gpt_page_2': chat_gpt_page_2,
            'chat_gpt_page_3': chat_gpt_page_3,
            'chat_gpt_page_4': chat_gpt_page_4,
            'chat_gpt_page_5': chat_gpt_page_5,
            'chat_gpt_page_6': chat_gpt_page_6,
            },
        )
    else:
        report_context.update(
            {
                'context_for_file': {
                    'chat_gpt_page_1': chat_gpt_page_1,
                    'chat_gpt_page_2': chat_gpt_page_2,
                    'chat_gpt_page_3': chat_gpt_page_3,
                    'chat_gpt_page_4': chat_gpt_page_4,
                    'chat_gpt_page_5': chat_gpt_page_5,
                    'chat_gpt_page_6': chat_gpt_page_6,
                },
            },
        )

    report.context = report_context
    report.save()

from google_sheets_parser.google_sheets_parser import GSheetsParser
from routines.dataframe_routines import pretty_dataframe
from routines.db_routines import select_campaign_metrics
import seaborn as sns
from io import BytesIO
import pandas as pd
from matplotlib import pyplot as plt
import openai


class Reporter:
    def __init__(self, sheet_link):
        self.parser = GSheetsParser(sheet_link)
        self.user_timeseries = pretty_dataframe(pd.DataFrame(self.parser.parse()))
        self.history_timeseries_for_this = select_campaign_metrics(drug_name=self.user_timeseries.drug_name.iloc[0],
                                                                   medic_group=self.user_timeseries.medic_group.iloc[0],
                                                                   adv_format=self.user_timeseries.adv_format.iloc[0])
        self.history_timeseries = select_campaign_metrics(drug_name=None, medic_group=None, adv_format=None)

    def validate_report(self, threshold=15):
        if abs(self.user_timeseries['moving_average_change'].tail(1).values[0]) > threshold:
            return True
        return False

    def generate_report_text(self):  # Текст отчёта
        if (self.user_timeseries.trend.tail(1).values[0] == 'Increase') & (
                self.user_timeseries.moving_average_change.tail(1).values[0] > 0):
            trend = f'''\nПоказатель CTR за последние дни вырос на {self.user_timeseries.moving_average_change.tail(1).values[0]:.2f}%!\nДля получения дополнительно информации, можете ознакомиться с графиками.  
            '''
        elif (self.user_timeseries.trend.tail(1).values[0] == 'Decrease') & (
                self.user_timeseries.moving_average_change.tail(1).values[0] < 0):
            trend = f'''\nПоказатель CTR за последние дни упал на {self.user_timeseries.moving_average_change.tail(1).values[0]:.2f}%.\nРекомендуется ознакомиться с графиком общей тенденции CTR, чтобы принять решение.\n
            '''
        else:
            trend = f'''\nПоказатель CTR колеблется, либо остаётся на одном уровне.\nОзнакомьтесь с другими графиками, чтобы принять взвешенное решение.
            '''
        impressions = self.user_timeseries.impressions.sum()
        ctr = self.user_timeseries.clicks.sum() / self.user_timeseries.impressions.sum()
        return (
            f"Отчёт по креативу:\n"
            f"Название препарата: {self.user_timeseries.drug_name[1]}\n"
            f"Целевая группа: {self.user_timeseries.medic_group[1]}\n"
            f"Тип креатива: {self.user_timeseries.adv_format[1]}\n"
            f"{trend}\n"
            f"Общее количество показов: {impressions}, это больше чем у "
            f"{(self.history_timeseries_for_this.groupby(['campaign_name']).impressions.sum() < impressions).sum() * 100 / (self.history_timeseries_for_this.groupby(['campaign_name']).impressions.sum() < impressions).count():.1f}% "
            f"кампаний для данных препарата/ца/формата креатива, а так же больше чем у "
            f"{(self.history_timeseries.groupby(['campaign_name']).impressions.sum() < impressions).sum() * 100 / (self.history_timeseries.groupby(['campaign_name']).impressions.sum() < impressions).count():.1f}% "
            f"всех предыдущих рекламных кампаний.\n\n"
            f"СTR кампании за всё время составил {ctr * 100:.3f}%. Для данных препарата/ца/формата креатива средний показатель CTR составляет "
            f"{(self.history_timeseries_for_this.groupby(['campaign_name']).clicks.sum() * 100 / self.history_timeseries_for_this.groupby(['campaign_name']).impressions.sum()).median():.3f}% "
            f"Средний показатель CTR для всех предыдущих рекламных кампаний составляет "
            f"{(self.history_timeseries.groupby(['campaign_name']).clicks.sum() * 100 / self.history_timeseries.groupby(['campaign_name']).impressions.sum()).median():.3f}%\n\n"
            f"На данный момент кампания длится {self.user_timeseries.day.iloc[-1]} дней."
            f"Обычно кампания для данных препарата/ца/формата креатива длится "
            f"{self.history_timeseries_for_this.groupby(['campaign_name', 'campaign_number'])['day'].max().median():.0f} дня/дней, а "
            f"средняя длительность всех рекламных кампаний составляет {self.history_timeseries.groupby(['campaign_name', 'campaign_number'])['day'].max().median():.0f} дня/дней.\n\n"
            f"В ходе этой кампании значение CTR росло {(self.user_timeseries.trend == 'Increase').sum()} дня/дней, падало {(self.user_timeseries.trend == 'Decrease').sum()} дня/дней и стабилизровалось, либо было стабильно {(self.user_timeseries.trend == 'Plateau').sum()} дня/дней."
        )

    def generate_report_images(self):
        report_images = []

        buffer = BytesIO()
        fig2, ax = plt.subplots(figsize=(15, 9))
        image2 = sns.lineplot(data=self.user_timeseries[self.user_timeseries['day'] > 5], x='day', y='rolled_5', ax=ax,
                              label="Текущая кампания")
        history = select_campaign_metrics(drug_name=None, medic_group=self.user_timeseries.medic_group.iloc[0],
                                          adv_format=None)
        history = history[history['click_rate'] < 0.015]
        history = history[(history['day'] > 5) & (history['day'] < self.user_timeseries.day.max() + 5)]
        image2 = sns.lineplot(data=history, x='day', y='rolled_5', ax=ax, label="Усредненная история кампаний")
        image2.set_ylabel("Среднее CTR")
        image2.set_xlabel("День кампании")
        image2.set_title("Сравнение среднего CTR для данной ГРУППЫ ВРАЧЕЙ в ходе прошлых кампаниях и нынешней")

        ax.legend()

        ax.grid(True)
        sns.set_style("whitegrid")

        image2.get_figure().savefig(buffer, format='png')

        buffer.seek(0)
        report_images.append(buffer)
        buffer.flush()

        buffer2 = BytesIO()
        fig2, ax2 = plt.subplots(figsize=(15, 9))
        image2 = sns.lineplot(data=self.user_timeseries[self.user_timeseries['day'] > 5], x='day', y='rolled_5', ax=ax2,
                              label="Текущая кампания")
        history = select_campaign_metrics(drug_name=None, medic_group=None, adv_format=None)
        history = history[history['click_rate'] < 0.015]
        history = history[(history['day'] > 5) & (history['day'] < self.user_timeseries.day.max() + 5)]
        image2 = sns.lineplot(data=history, x='day', y='rolled_5', ax=ax2, label="Усредненная история кампаний")
        image2.set_ylabel("Среднее CTR")
        image2.set_xlabel("День кампании")
        image2.set_title("Сравнение среднего CTR для данного ТИПА КРЕАТИВА в ходе прошлых кампаниях и нынешней")

        ax2.legend()

        ax2.grid(True)
        sns.set_style("whitegrid")

        image2.get_figure().savefig(buffer2, format='png')

        buffer2.seek(0)
        report_images.append(buffer2)
        buffer2.flush()

        buffer3 = BytesIO()
        fig3, ax3 = plt.subplots(figsize=(15, 9))
        image3 = sns.lineplot(data=self.user_timeseries[self.user_timeseries['day'] > 5], x='day', y='rolled_5', ax=ax3,
                              label="Текущая кампания")
        history = select_campaign_metrics(drug_name=self.user_timeseries.drug_name.iloc[0], medic_group=None,
                                          adv_format=None)
        history = history[history['click_rate'] < 0.015]
        history = history[(history['day'] > 5) & (history['day'] < self.user_timeseries.day.max() + 5)]
        image3 = sns.lineplot(data=history, x='day', y='rolled_5', ax=ax3, label="Усредненная история кампаний")
        image3.set_ylabel("Среднее CTR")
        image3.set_ylabel("День кампании")
        image2.set_title("Информация об изменении CTR в ходе кампании")
        image3.set_title("Сравнение среднего CTR данного ПРЕПАРАТА в ходе прошлых кампаниях и нынешней")
        ax3.legend()

        ax3.grid(True)
        sns.set_style("whitegrid")

        image3.get_figure().savefig(buffer3, format='png')

        buffer3.seek(0)
        report_images.append(buffer3)
        buffer3.flush()

        return report_images

    @staticmethod
    def __get_chat_gpt_token():
        token_file = open('../tokens/chat_gpt_token', 'r')
        token = token_file.read()
        token_file.close()
        return token

    def generate_gpt_recommendation(self, report):
        openai.api_key = self.__get_chat_gpt_token()

        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {
                    "role": "system",
                    "content": "Ты - полезный помощник, который дает рекомендации, основанные на отчете о текущей маркетинговой эффективности баннера медицинского препарата.  Например, сформируй несколько актуальных гипотез относительно этих данных, приведи за и против каждой и постарайся сделать один целостный вывод относительно всех входных данных:Как этот рекламный материал себя показывает, и что следует сделать с этой кампанией дальше: Сворачивать и начинать другую, продолжать закупать столько же просмотров, или увеличивать обороты. Также постарайся не рекомендовать изучать эти данные дальше, сделай вид, что это вся информация, которую можно получить. тренд показателя CTR в этих данных может часто меняться, поэтому считай, что падение или рост показателя CTR действительно имеет значение, когда оно больше 10-15%. Если падение или рост меньше 10-15%, говори об этом только как о нейтральном показателе, делай выводы только если рост или падение больше этого значения. Если кампания подходит к среднему количеству дней у других кампаний, реши, продолжать её или закончить, в зависимости от показателя ctr и тренда. Также обращай внимание на то, сколько дней кампания растёт. Большое или малое количество показов по сути просто говорит, сколько денег уже было заплачено за эту рекламу, этот показатель в основном отвечает за это. Твои гипотезы должны касаться двух основных направлений: Как CTR кампании ведёт себя с течением времени и как CTR кампании сопоставляется средним данным."
                },
                {
                    "role": "user",
                    "content": report
                }
            ]
        )

        return response['choices'][0]['message']['content']


def send_report(bot, chat_id, sheet_link, validation=False):
    reporter = Reporter(sheet_link)
    gpt_recommendation = reporter.generate_gpt_recommendation(reporter.generate_report_text())
    if validation or reporter.validate_report():
        bot.send_message(chat_id, reporter.generate_report_text())
        bot.send_message(chat_id, gpt_recommendation)
        for current_image in reporter.generate_report_images():
            bot.send_photo(chat_id, photo=current_image)

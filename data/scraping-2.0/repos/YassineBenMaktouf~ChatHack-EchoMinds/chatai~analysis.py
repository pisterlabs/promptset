# analysis_app/analysis.py

import os
from django.conf import settings
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
import base64
from retrying import retry
import openai

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_chatgpt_analysis(query):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=query,
            max_tokens=100
        )
        if response and 'choices' in response and response['choices']:
            return response['choices'][0]['text'].strip()
        else:
            return "No response or unexpected format from AI."
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return f"AI encountered an error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"Unexpected error occurred: {str(e)}"


def convert_image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"


def insert_image_to_excel(worksheet, img_path, img_cell, width=None, height=None):
    img = Image(img_path)
    if width and height:
        img.width, img.height = width, height
    worksheet.add_image(img, img_cell)
def perform_analysis(file_path):
    df = pd.read_excel(file_path)

    # Your analysis logic goes here...
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['net'] = df['income'] - df['expenses']
    df['cumulative'] = df['net'].cumsum()

    fig_cumulative, ax_cumulative = plt.subplots(figsize=(8, 4))
    sns.lineplot(x='date', y='cumulative', data=df, ax=ax_cumulative, marker='o')
    ax_cumulative.axhline(y=0, color='red', linestyle='--')
    plt.tight_layout()
    plot_img_path = 'cumulative_plot.png'
    fig_cumulative.savefig(plot_img_path)
    plt.close(fig_cumulative)
    plot_url = convert_image_to_base64(plot_img_path)

    category_summary = df.groupby('category').agg({'net': 'sum'})
    category_summary_positive = category_summary[category_summary['net'] > 0]

    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    category_summary_positive.plot.pie(y='net', ax=ax_pie, autopct='%1.1f%%', startangle=140, legend=False)
    plt.tight_layout()
    pie_img_path = 'pie_chart.png'
    fig_pie.savefig(pie_img_path)
    plt.close(fig_pie)
    pie_chart_url = convert_image_to_base64(pie_img_path)

    df['year'] = df['date'].dt.year
    yearly_summary = df.groupby('year').agg({'income': 'sum', 'expenses': 'sum', 'net': 'sum'})
    yearly_summary['win_percentage'] = np.where(yearly_summary['net'] > 0,
                                                yearly_summary['net'] / yearly_summary['income'] * 100, 0)
    summary_html = yearly_summary.to_html(classes="table table-striped", float_format='%.2f')

    extracted_insights = """
        - The company's profits have consistently increased over the past five years.
        - The profit margin has improved due to cost-cutting measures.
        - There was a significant boost in revenue from a new product launch.
    """

    query = f"Provide a financial analysis summary and suggestions for improvement based on the following data: {extracted_insights}"
    ai_analysis = None #get_chatgpt_analysis(query)

    # Construct a new file path with a unique name
    analyzed_file_name = f'analyzed_{os.path.basename(file_path)}'
    analyzed_file_path = os.path.join(settings.MEDIA_ROOT, analyzed_file_name)

    # Save the analyzed data to the new file path
    wb = Workbook()
    ws_data = wb.active
    ws_data.title = "Financial Data"
    for r in dataframe_to_rows(df, index=False, header=True):
        ws_data.append(r)
    insert_image_to_excel(ws_data, plot_img_path, 'A10')
    insert_image_to_excel(ws_data, pie_img_path, 'A40')
    wb.save(analyzed_file_path)

    return {
        'plot_url': plot_url,
        'pie_chart_url': pie_chart_url,
        'summary_html': summary_html,
        'ai_analysis': ai_analysis,
        'file_url': f'{settings.MEDIA_URL}{analyzed_file_name}',  # Use the new file name
    }
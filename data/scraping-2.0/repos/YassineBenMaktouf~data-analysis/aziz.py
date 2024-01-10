import requests
import os
from flask import Flask, jsonify, request, render_template_string, send_file
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import io
import numpy as np
from dotenv import load_dotenv
import base64
from retrying import retry
import openai

app = Flask(__name__)
sns.set(style="whitegrid")

html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Excel File Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        h1 { color: #333; }
        form { margin-bottom: 20px; }
        input[type="file"] { display: block; margin: 10px auto; }
        input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        input[type="submit"]:hover { background-color: #45a049; }
        .container { width: 80%; margin: auto; }
        img { margin-top: 20px; max-width: 49%; display: inline-block; }
        a { background-color: #008CBA; color: white; padding: 10px 20px; text-decoration: none; display: inline-block; }
        a:hover { background-color: #005f6a; }
        .summary { background-color: #f2f2f2; padding: 10px; border-radius: 5px; margin-top: 20px; }
        .ai-analysis { border: 1px solid #ddd; background-color: #f9f9f9; padding: 20px; margin-top: 20px; text-align: left; }
        .chat-container { display: none; position: fixed; bottom: 10px; right: 10px; background: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
        #chat-box { height: 200px; overflow-y: auto; margin-bottom: 10px; }
        #chat-input { width: calc(100% - 110px); }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload Excel File for Analysis</h1>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".xlsx, .xls" required>
            <input type="submit" value="Analyze">
        </form>
        <div>
            {% if plot_url %}
                <img src="{{ plot_url }}" alt="Cumulative Profit/Loss Plot">
            {% endif %}
            {% if pie_chart_url %}
                <img src="{{ pie_chart_url }}" alt="Financial Distribution">
            {% endif %}
        </div>
        {% if summary %}
            <div class="summary">
                <h2>Yearly Summary:</h2>
                <p>{{ summary|safe }}</p>
            </div>
        {% endif %}
        {% if ai_analysis %}
            <div class="ai-analysis">
                <h2>AI-Driven Analysis:</h2>
                <p>{{ ai_analysis|safe }}</p>
            </div>
        {% endif %}
        {% if file_url %}
            <a href="{{ file_url }}" download="analyzed_data.xlsx">Download Analyzed Data</a>
        {% endif %}
    </div>
    <div class="chat-container" id="chat-container">
        <div id="chat-box"></div>
        <input type="text" id="chat-input" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>
    </div>
    <button onclick="toggleChat()">Chat with AI</button>
    <script type="text/javascript">
        function toggleChat() {
            var chatContainer = document.getElementById('chat-container');
            chatContainer.style.display = chatContainer.style.display === 'none' ? 'block' : 'none';
        }
        function sendMessage() {
            var input = document.getElementById('chat-input');
            var message = input.value;
            input.value = '';
            if(message) {
                document.getElementById('chat-box').innerHTML += '<div>You: ' + message + '</div>';
                fetch('/chat', {
                    method: 'POST',
                    body: JSON.stringify({'message': message}),
                    headers: {'Content-Type': 'application/json'}
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('chat-box').innerHTML += '<div>AI: ' + data.response + '</div>';
                });
            }
        }
    </script>
</body>
</html>
'''

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
        if response:
            return response.choices[0].text
        else:
            return "Analysis not available"
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Analysis not available"

def convert_image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

def insert_image_to_excel(worksheet, img_path, img_cell, width=None, height=None):
    img = Image(img_path)
    if width and height:
        img.width, img.height = width, height
    worksheet.add_image(img, img_cell)

@app.route('/')
def index():
    return render_template_string(html)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        df = pd.read_excel(file)
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
        yearly_summary['win_percentage'] = np.where(yearly_summary['net'] > 0, yearly_summary['net'] / yearly_summary['income'] * 100, 0)
        summary_html = yearly_summary.to_html(classes="table table-striped", float_format='%.2f')
        extracted_insights = """
        - The company's profits have consistently increased over the past five years.
        - The profit margin has improved due to cost-cutting measures.
        - There was a significant boost in revenue from a new product launch.
        """
        query = f"Provide a financial analysis summary and suggestions for improvement based on the following data: {extracted_insights}"
        ai_analysis = get_chatgpt_analysis(query)
        analyzed_file_path = 'analyzed_financial_data.xlsx'
        wb = Workbook()
        ws_data = wb.active
        ws_data.title = "Financial Data"
        for r in dataframe_to_rows(df, index=False, header=True):
            ws_data.append(r)
        insert_image_to_excel(ws_data, plot_img_path, 'A10')
        insert_image_to_excel(ws_data, pie_img_path, 'A40')
        wb.save(analyzed_file_path)
        os.remove(plot_img_path)
        os.remove(pie_img_path)
        file_url = '/download/' + analyzed_file_path
        return render_template_string(html, plot_url=plot_url, pie_chart_url=pie_chart_url, summary=summary_html, ai_analysis=ai_analysis, file_url=file_url)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    ai_response = get_chatgpt_analysis(user_message)
    return jsonify({'response': ai_response})


if __name__ == '__main__':
    app.run(debug=True)

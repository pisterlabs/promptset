import os
import openai
from flask import Flask, render_template_string, request
from googleapiclient.discovery import build

# 環境変数からAPIキーとカスタム検索エンジンID、OpenAI APIキーを取得
API_KEY = os.environ.get('GOOGLE_API_KEY')
CUSTOM_SEARCH_ENGINE_ID = os.environ.get('CUSTOM_SEARCH_ENGINE_ID')
OPENAI_API_SECRET_KEY = os.environ.get('OPENAI_API_SECRET_KEY')

app = Flask(__name__)

# 検索結果の情報を整理するメソッド（日付情報の追加）
def summarize_search_results_with_date(items):
    result_items = []
    for item in items:
        date = item.get('pagemap', {}).get('metatags', [{}])[0].get('date', 'Unknown Date')
        result_items.append({
            'title': item['title'],
            'snippet': item['snippet'],
            'date': date,
            'url': item['link']
        })
    return result_items


# OpenAIのAPIを利用して複数の検索結果間で重複を判定する関数（日付情報を含む）
def find_duplicates_with_date(search_results):
    openai.api_key = OPENAI_API_SECRET_KEY
    prompt = "This is a dataset of news articles with their titles, snippets, and dates. Divide all articles into a few groups based on their similarities.\n\n"

    # 検索結果の情報を組み合わせる
    for i in range(len(search_results)):
        date_i = search_results[i].get('date', 'Unknown Date')
        prompt += f"Article {i+1}: Title: \"{search_results[i]['title']}\", Snippet: \"{search_results[i]['snippet']}\", Date: \"{date_i}\"\n"

    prompt += "\nAre any of the above articles duplicates of each other? If yes, list the article numbers that are duplicates."

    # OpenAI APIを呼び出して重複を判定
    response = openai.Completion.create(
        model="gpt-3.5-turbo-1106",
        prompt=prompt,
        max_tokens=2000
    )

    # 応答を解析する
    answers = response['choices'][0]['text'].strip().lower()
    duplicates = []
    if "no" not in answers:
        # 応答から数字のペアを抽出する処理
        lines = answers.split("\n")
        for line in lines:
            if "article" in line:
                indices = [int(word) - 1 for word in line.split() if word.isdigit()]
                duplicates.extend(indices)

    return duplicates



def remove_duplicates_with_date_improved(search_results):
    duplicates = find_duplicates_with_date(search_results)
    
    # 重複があるインデックスを記録
    duplicate_indices_to_remove = set()
    for i, j in duplicates:
        # 既に重複リストにない場合、2番目の要素を削除リストに追加
        if i not in duplicate_indices_to_remove:
            duplicate_indices_to_remove.add(j)

    # 重複がない結果のみを保持する
    unique_results = [result for i, result in enumerate(search_results) if i not in duplicate_indices_to_remove]

    return unique_results



# APIにアクセスして結果を取得するメソッド
def get_search_results(query):
    search = build("customsearch", "v1", developerKey=API_KEY)
    # 最初の10件の検索結果を取得
    result1 = search.cse().list(q=query, cx=CUSTOM_SEARCH_ENGINE_ID, lr='lang_en', num=3, start=1).execute()
    items1 = result1.get('items', [])

    # 次の10件の検索結果を取得
    result2 = search.cse().list(q=query, cx=CUSTOM_SEARCH_ENGINE_ID, lr='lang_en', num=1, start=11).execute()
    items2 = result2.get('items', [])

    # 両方の結果を結合
    return items1 + items2
# ⭐︎numを１０に変更すること！



# 検索結果の情報を整理するメソッド
def summarize_search_results(items):
    result_items = []
    for item in items:
        result_items.append({
            'title': item['title'],
            'url': item['link'],
            'snippet': item['snippet']
        })
    return result_items

@app.route('/', methods=['GET', 'POST'])
def index():
    raw_search_results = []
    unique_search_results = []  
    if request.method == 'POST':
        keyword1 = request.form.get('keyword1', '')
        keyword2 = request.form.get('keyword2', '')
        keyword3 = request.form.get('keyword3', '')
        period = request.form.get('period', 'all')

        # 新しい期間オプションに基づいてクエリを設定
        if period == '1day':
            combined_query = f"\"{keyword1}\" AND \"{keyword2}\" AND \"{keyword3}\" after:1d".strip()
        elif period == '7days':
            combined_query = f"\"{keyword1}\" AND \"{keyword2}\" AND \"{keyword3}\" after:7d".strip()
        elif period == '1month':
            combined_query = f"\"{keyword1}\" AND \"{keyword2}\" AND \"{keyword3}\" after:1m".strip()
        elif period == '3months':
            combined_query = f"\"{keyword1}\" AND \"{keyword2}\" AND \"{keyword3}\" after:3m".strip()
        elif period == '6months':
            combined_query = f"\"{keyword1}\" AND \"{keyword2}\" AND \"{keyword3}\" after:6m".strip()
        elif period == '12months':
            combined_query = f"\"{keyword1}\" AND \"{keyword2}\" AND \"{keyword3}\" after:12m".strip()
        else:
            combined_query = f"\"{keyword1}\" AND \"{keyword2}\" AND \"{keyword3}\"".strip()

        raw_results = get_search_results(combined_query)
        raw_search_results = summarize_search_results(raw_results)
        unique_search_results = remove_duplicates_with_date_improved(raw_search_results)

    return render_template_string('''
<!doctype html>
<html>
<head>
    <title>Search with Google</title>
    <style>
        #loading {
            display: none;
            color: blue;
            font-size: 20px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <form id="searchForm" method="post">
        Keyword 1: <input type="text" name="keyword1"><br>
        Keyword 2: <input type="text" name="keyword2"><br>
        Keyword 3: <input type="text" name="keyword3"><br>
        Period:
        <select name="period">
            <option value="all">All Periods</option>
            <option value="1day">Last 1 Day</option>
            <option value="7days">Last 7 Days</option>
            <option value="1month">Last 1 Month</option>
            <option value="3months">Last 3 Months</option>
            <option value="6months">Last 6 Months</option>
            <option value="12months">Last 12 Months</option>
        </select><br>
        <input type="submit" value="Search">
    </form>

    <div id="loading">loading...</div>

    <h2>Unique Results (Duplicates Removed)</h2>
    <div style="border: 1px solid #ddd; margin-bottom: 20px;">
        {% if unique_search_results %}
            <ul>
            {% for item in unique_search_results %}
                <li><a href="{{ item.url }}">{{ item.title }}</a> <br>- {{ item.snippet }}</li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No unique results found.</p>
        {% endif %}
    </div>

    <h2>All Results (Before Removing Duplicates)</h2>
    <div style="border: 1px solid #ddd;">
        {% if raw_search_results %}
            <ul>
            {% for item in raw_search_results %}
                <li><a href="{{ item.url }}">{{ item.title }}</a> <br>- {{ item.snippet }}</li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No results found.</p>
        {% endif %}
    </div>

    <script>
        document.getElementById('searchForm').onsubmit = function() {
            document.getElementById('loading').style.display = 'block';
        };
    </script

</body>
</html>

    ''', unique_search_results=unique_search_results, raw_search_results=raw_search_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

import evadb
import openai
import os
import csv
import re
import json
import nest_asyncio
import numpy as np
"""
This script is made to provide a e2e experience, which takes in a directory of articles and a CSV of categories, and outputs the most relevant category for each article.
The logic is largely the same as in report.ipynb.
However due to hardware constraints, I could not test this script locally.
"""


class Categorizer:
    def __init__(self, api_key, category_csv_path, article_dir_path):
        openai.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        path = os.path.dirname(evadb.__file__)
        self.cursor = evadb.connect(path).cursor()
        self.api_key = api_key
        # self.cursor = evadb.connect().cursor()
        self.category_csv_path = category_csv_path
        self.article_dir_path = article_dir_path

    def cache_summary_exists(self, article_id):
        """Check if the summary cache exists for a given article."""
        cache_file = os.path.join('summary_cache', f'{article_id}.json')
        return os.path.exists(cache_file)

    def read_cached_summary(self, article_id):
        """Read and return the cached summary."""
        with open(os.path.join('summary_cache', f'{article_id}.json'), 'r') as file:
            return json.load(file)

    def cache_summary(self, article_id, summary):
        """Cache the summary."""
        os.makedirs('summary_cache', exist_ok=True)
        with open(os.path.join('summary_cache', f'{article_id}.json'), 'w') as file:
            json.dump(summary, file)

    def populate_articles_table(self):
        self.cursor.query("""
            DROP TABLE IF EXISTS articles
        """).df()
        self.cursor.query("""
            CREATE TABLE articles (id INTEGER, article TEXT(30000))
        """).df()

        texts = []
        for filename in os.listdir(self.article_dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.article_dir_path, filename), 'r') as file:
                    text = file.read().replace("\n", " ")
                    text = re.sub(r'[^A-Za-z ]', '', text)
                    texts.append(text)

        # Create a temporary table for summaries
        self.cursor.query("""DROP TABLE IF EXISTS temp_summaries""").df()
        self.cursor.query("""
            CREATE TABLE temp_summaries (id INTEGER, summary_text TEXT)
        """).df()

        for i, t in enumerate(texts):
            self.cursor.query(
                f"INSERT INTO articles (id, article) VALUES ({i}, '{t}')").df()

            # Check for cached summary
            if self.cache_summary_exists(i):
                summary = self.read_cached_summary(i)
            else:
                self.cursor.query("""
                CREATE FUNCTION IF NOT EXISTS TextSummarizer
                TYPE HuggingFace
                TASK 'summarization'
                MODEL 'facebook/bart-large-cnn';
                """).df()
                # If no cache, generate summary and cache it
                summary = self.cursor.query(
                    f"SELECT TextSummarizer('{t}')").df().iloc[0, 0]
                self.cache_summary(i, summary)

            # Insert summary into temporary table
            self.cursor.query(
                f"INSERT INTO temp_summaries (id, summary_text) VALUES ({i}, '{summary}')").df()

        # Create the articles_with_summaries table
        self.cursor.query("""
            DROP TABLE IF EXISTS articles_with_summaries
        """).df()

        self.cursor.query("""
            CREATE TABLE articles_with_summaries AS
            SELECT a.id, a.article, t.summary_text
            FROM articles AS a
            JOIN temp_summaries AS t ON a.id = t.id
        """).df()

    def get_kth_level_categories(self, k):
        """
        Reads a CSV file and extracts k-th level categories.

        :param category_csv_path: Path to the CSV file containing the categories.
        :param k: The level of category to extract (1-based index).
        :return: A set containing the k-th level categories.
        """
        kth_level_categories = set()
        with open(self.category_csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                # Check if the row has enough columns for the k-th level category
                if len(row) >= k:
                    # Get the k-th level category, making sure it's not empty
                    # k-1 because list index is 0-based
                    category = row[k-1].strip()
                    if category:
                        kth_level_categories.add(category)
        return kth_level_categories

    def populate_categories_table(self):
        def filter_english_chars(text):
            return re.sub(r'[^A-Za-z ]', '', text)

        self.cursor.query("DROP TABLE IF EXISTS categories").df()
        self.cursor.query(
            "CREATE TABLE categories (id INTEGER, category TEXT(30))").df()

        nest_asyncio.apply()
        third_level_categories = self.get_kth_level_categories(3)
        categories_list = list(third_level_categories)
        filtered_categories = sorted(
            [filter_english_chars(category) for category in categories_list])[:30]

        for idx, category in enumerate(filtered_categories):
            self.cursor.query(
                f"INSERT INTO categories (id, category) VALUES ({idx}, '{category}')").df()

    def populate_articles_table(self):
        self.cursor.query("""
            DROP TABLE IF EXISTS articles
        """).df()
        self.cursor.query("""
            CREATE TABLE articles (id INTEGER, article TEXT(30000))
        """).df()

        texts = []
        for filename in os.listdir(self.article_dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.article_dir_path, filename), 'r') as file:
                    text = file.read().replace("\n", " ")
                    text = re.sub(r'[^A-Za-z ]', '', text)
                    texts.append(text)

        self.cursor.query("""DROP TABLE IF EXISTS temp_summaries""").df()
        # Create a temporary table for summaries
        self.cursor.query("""
            CREATE TABLE temp_summaries (id INTEGER, summary_text TEXT)
        """).df()

        for i, t in enumerate(texts):
            self.cursor.query(
                f"INSERT INTO articles (id, article) VALUES ({i}, '{t}')").df()

            # Check for cached summary
            if self.cache_summary_exists(i):
                summary = self.read_cached_summary(i)
            else:
                # If no cache, generate summary and cache it
                summary = self.cursor.query(
                    f"SELECT TextSummarizer('{t}')").df().iloc[0, 0]
                self.cache_summary(i, summary)

            # Insert summary into temporary table
            self.cursor.query(
                f"INSERT INTO temp_summaries (id, summary_text) VALUES ({i}, '{summary}')").df()

        # Create the articles_with_summaries table
        self.cursor.query("""
            DROP TABLE IF EXISTS articles_with_summaries
        """).df()

        self.cursor.query("""
            CREATE TABLE articles_with_summaries AS
            SELECT a.id, a.article, t.summary_text
            FROM articles AS a
            JOIN temp_summaries AS t ON a.id = t.id
        """).df()

    def _old_populate_articles_table(self):
        self.cursor.query("""
            DROP TABLE IF EXISTS articles
        """).df()
        self.cursor.query("""
            CREATE TABLE articles (id INTEGER, article TEXT(30000))
        """).df()

        texts = []
        for filename in os.listdir(self.article_dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.article_dir_path, filename), 'r') as file:
                    text = file.read().replace("\n", " ")
                    text = re.sub(r'[^A-Za-z ]', '', text)
                    texts.append(text)

        for i, t in enumerate(texts):
            self.cursor.query(
                f"INSERT INTO articles (id, article) VALUES ({i}, '{t}')").df()

        self.cursor.query("""
        CREATE FUNCTION IF NOT EXISTS TextSummarizer
        TYPE HuggingFace
        TASK 'summarization'
        MODEL 'facebook/bart-large-cnn';
        """).df()

        self.cursor.query(f"""
        DROP TABLE IF EXISTS temp_summaries;
        """).df()

        self.cursor.query(f"""
        CREATE TABLE temp_summaries AS
        SELECT id, TextSummarizer(article)
        FROM articles;
        """).df()

        self.cursor.query("""
        DROP TABLE IF EXISTS articles_with_summaries;
        """).df()

        self.cursor.query("""
        CREATE TABLE articles_with_summaries AS
        SELECT a.id, a.article, t.summary_text
        FROM articles AS a
        JOIN temp_summaries AS t ON a.id = t.id;
        """).df()

    def execute_matching(self):
        self.cursor.query(
            "DROP FUNCTION IF EXISTS OpenAIEmbeddingExtractor;").df()
        self.cursor.query(f"""
        CREATE FUNCTION IF NOT EXISTS OpenAIEmbeddingExtractor
        IMPL './openai_embedding_extractor.py';
        """).df()
        self.cursor.query("""
            CREATE INDEX index_table
            ON categories (OpenAIEmbeddingExtractor(category))
            USING FAISS;
        """).df()

        # Create the result table structure
        self.cursor.query("""
        DROP TABLE IF EXISTS article_similar_categories;
        """).df()

        self.cursor.query("""
        CREATE TABLE article_similar_categories (
            article_id INTEGER,
            summary TEXT(3000),
            category_1 TEXT(100),
            category_2 TEXT(100),
            category_3 TEXT(100),
            category_4 TEXT(100),
            category_5 TEXT(100)
        );
        """).df()

        all_articles = self.cursor.query(
            "SELECT id, summary_text FROM articles_with_summaries;").df()

        for index, row in all_articles.iterrows():
            article_id = row[0]
            summary_text = row[1]

            similar_categories = self.cursor.query(f"""
            SELECT category FROM categories
            ORDER BY Similarity(
                OpenAIEmbeddingExtractor('{summary_text}'),
                OpenAIEmbeddingExtractor(category)
            )
            LIMIT 5;
            """).df()

            # Extracting top 5 categories. If there are fewer than 5 results, the rest will be set as None.
            cat_1 = similar_categories.iloc[0][0] if len(
                similar_categories) > 0 else None
            cat_2 = similar_categories.iloc[1][0] if len(
                similar_categories) > 1 else None
            cat_3 = similar_categories.iloc[2][0] if len(
                similar_categories) > 2 else None
            cat_4 = similar_categories.iloc[3][0] if len(
                similar_categories) > 3 else None
            cat_5 = similar_categories.iloc[4][0] if len(
                similar_categories) > 4 else None

            # Insert the results into the new table
            self.cursor.query(f"""
            INSERT INTO article_similar_categories (article_id, summary, category_1, category_2, category_3, category_4, category_5)
            VALUES ({article_id}, '{summary_text}', '{cat_1}', '{cat_2}', '{cat_3}', '{cat_4}', '{cat_5}');
            """).df()

    def refine_matches_with_chatgpt(self):
        self.cursor.query("DROP TABLE IF EXISTS article_final_category;").df()
        self.cursor.query("""
        CREATE TABLE article_final_category (
            article_id INTEGER,
            selected_category TEXT(300)
        );
        """).df()

        all_articles = self.cursor.query(
            "SELECT * FROM article_similar_categories;").df()

        for index, row in all_articles.iterrows():
            article_id = row[1]
            summary_text = row[2]
            categories = [row[3], row[4], row[5], row[6], row[7]]

            prompt = (f"Given the summary: '{summary_text}', "
                      f"please choose the category that most closely aligns with the topic. DO NOT OUTPUT A SENTENCE, JUST THE CATEGORY"
                      f"If none are even remotely related, reply 'none'. "
                      f"The available categories are: {', '.join([cat for cat in categories if cat])}.")

            category_choice = self.cursor.query(f"""
            SELECT ChatGPT("{prompt}")
            """).df().iloc[0][0]

            # Insert the result into the new table
            self.cursor.query(f"""
            INSERT INTO article_final_category (article_id, selected_category)
            VALUES ({article_id}, '{category_choice}');
            """).df()

    def execute_query(self, query):
        print(self.cursor.query(query).df())
